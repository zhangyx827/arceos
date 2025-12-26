use alloc::{
    boxed::Box,
    collections::btree_map::BTreeMap,
    string::String,
    sync::Arc,
    vec::Vec,
};
use core::slice;

use axerrno::{AxError, AxResult};
use axfs_ng::FileBackend;
use axfs_ng_vfs::{VfsError, VfsResult};
use axhal::{
    mem::phys_to_virt,
    paging::{MappingFlags, PageSize, PageTableMut, PagingError},
};
use axsync::Mutex;
use kspin::SpinNoIrq;
use lazy_static::lazy_static;
use memory_addr::{MemoryAddr, PAGE_SIZE_4K, PhysAddr, VirtAddr, VirtAddrRange};

use crate::{
    AddrSpace,
    THP_PAGE_BYTES,
    backend::{Backend, BackendOps, VmaFlags, alloc_frame, dealloc_frame, pages_in},
};

static FRAME_TABLE: SpinNoIrq<BTreeMap<PhysAddr, u8>> = SpinNoIrq::new(BTreeMap::new());

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThpPolicy {
    Always,
    Madvise,
    Never,
}

impl ThpPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Always => "always",
            Self::Madvise => "madvise",
            Self::Never => "never",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "always" => Some(Self::Always),
            "madvise" => Some(Self::Madvise),
            "never" => Some(Self::Never),
            _ => None,
        }
    }
}

// Global THP policy controlled through `/sys/kernel/mm/transparent_hugepage/enabled`.
lazy_static! {
    static ref GLOBAL_THP_POLICY: SpinNoIrq<ThpPolicy> = SpinNoIrq::new(ThpPolicy::Madvise);
}

/// Updates the global THP policy string (e.g. "always", "madvise", "never").
pub fn set_thp_policy(policy: &str) -> VfsResult<Vec<u8>> {
    // Some writers may perform a truncate-like write with an empty buffer before
    // writing the real contents (even though sysfs normally doesn't require
    // this). Treat empty writes as a no-op for robustness.
    let trimmed = policy.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let Some(policy) = ThpPolicy::parse(trimmed) else {
        return Err(VfsError::InvalidInput);
    };
    *GLOBAL_THP_POLICY.lock() = policy;
    Ok(Vec::new())
}

/// Returns the current global THP policy string 
pub fn current_thp_policy() -> String {
    GLOBAL_THP_POLICY.lock().as_str().into()
}

/// Returns the approximate reference count for a given frame managed by COW.
///
/// This is only meaningful for frames tracked in `FRAME_TABLE`. For other
/// frames, it returns 0.
pub fn frame_ref_count(paddr: PhysAddr) -> u8 {
    FRAME_TABLE
        .lock()
        .get(&paddr)
        .copied()
        .unwrap_or(0)
}


fn inc_frame_ref(paddr: PhysAddr) {
    let mut table = FRAME_TABLE.lock();
    *table.entry(paddr).or_insert(0) += 1;
}

fn dec_frame_ref(paddr: PhysAddr) -> usize {
    let mut table = FRAME_TABLE.lock();
    if let Some(count) = table.get_mut(&paddr) {
        let prev = *count;
        if prev == 1 {
            table.remove(&paddr);
        } else {
            *count -= 1;
        }
        prev as usize
    } else {
        0
    }
}

/// Copy-on-write mapping backend.
///
/// This corresponds to the `MAP_PRIVATE` flag.
pub struct CowBackend {
    start: VirtAddr,
    size: PageSize,
    file: Option<(FileBackend, u64, Option<u64>)>,
    // Per-VMA THP policy flags (VM_HUGEPAGE / VM_NOHUGEPAGE).
    // Protected by the outer AddrSpace lock; we still use a Mutex here so that
    // the backend type remains Sync and can live inside KERNEL_ASPACE.
    vma_flags: Mutex<VmaFlags>,
}

impl Clone for CowBackend {
    fn clone(&self) -> Self {
        Self {
            start: self.start,
            size: self.size,
            file: self.file.clone(),
            // Each cloned backend gets its own mutex, but we preserve the
            // current THP policy bits.
            vma_flags: Mutex::new(*self.vma_flags.lock()),
        }
    }
}

impl CowBackend {
    fn alloc_new_at(
        &self,
        vaddr: VirtAddr,
        flags: MappingFlags,
        pt: &mut PageTableMut,
    ) -> AxResult {
        let frame = alloc_frame(true, self.size)?;
        inc_frame_ref(frame);

        if let Some((file, file_start, file_end)) = &self.file {
            let buf = unsafe {
                slice::from_raw_parts_mut(phys_to_virt(frame).as_mut_ptr(), self.size as _)
            };
            // vaddr can be smaller than self.start (at most 1 page) due to
            // non-aligned mappings, we need to keep the gap clean.
            let start = self.start.as_usize().saturating_sub(vaddr.as_usize());
            assert!(start < self.size as _);

            let file_start =
                *file_start + vaddr.as_usize().saturating_sub(self.start.as_usize()) as u64;
            let max_read = file_end
                .map_or(u64::MAX, |end| end.saturating_sub(file_start))
                .min((buf.len() - start) as u64) as usize;

            file.read_at(&mut &mut buf[start..start + max_read], file_start)?;
        }
        pt.map(vaddr, frame, self.size, flags)?;
        Ok(())
    }

    fn handle_cow_fault(
        &self,
        vaddr: VirtAddr,
        paddr: PhysAddr,
        flags: MappingFlags,
        pt: &mut PageTableMut,
    ) -> AxResult {
        match dec_frame_ref(paddr) {
            0 => unreachable!(),
            // There is only one AddrSpace reference to the page,
            // so there is no need to copy it.
            1 => {
                inc_frame_ref(paddr);
                pt.protect(vaddr, flags)?;
            }
            // Allocates the new page and copies the contents of the original page,
            // remapping the virtual address to the physical address of the new page.
            2.. => {
                let new_frame = alloc_frame(false, self.size)?;
                inc_frame_ref(new_frame);
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        phys_to_virt(paddr).as_ptr(),
                        phys_to_virt(new_frame).as_mut_ptr(),
                        self.size as _,
                    );
                }
                pt.remap(vaddr, new_frame, flags)?;
            }
        }
        Ok(())
    }
}

impl BackendOps for CowBackend {
    /// Collapse a fully-eligible 2 MiB window into a single huge page.
    fn collapse_page(
        &self,
        range: VirtAddrRange,
        pt: &mut PageTableMut,
        fault_in: bool,
        _callbacks: Option<&mut Vec<Box<dyn FnOnce(&mut AddrSpace)>>>,
    ) -> AxResult {
        let new_pa = alloc_frame(true, PageSize::Size2M)?;
        let mut flags_opt: Option<MappingFlags> = None;
        let mut fault_addrs = Vec::new();
        let mut old_pages = Vec::new();
        for page_va in pages_in(range, PageSize::Size4K)? {
            match pt.query(page_va) {
                Ok((old_pa, page_flags, _)) => {
                    if flags_opt.is_none() {
                        flags_opt = Some(page_flags);
                    }
                    let offset = page_va - range.start;
                    let src = phys_to_virt(old_pa);
                    let dst_pa = new_pa + offset;
                    let dst = phys_to_virt(dst_pa);
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            src.as_ptr(),
                            dst.as_mut_ptr(),
                            PAGE_SIZE_4K,
                        );
                    }
                    old_pages.push(old_pa);
                },
                Err(PagingError::NotMapped) => {
                    if fault_in {
                        fault_addrs.push(page_va);
                    }
                }
                _ => {
                    return Err(AxError::BadAddress);
                },
            }
        }

        let Some(flags) = flags_opt else {
            // All the pages are not mapped
            dealloc_frame(new_pa, PageSize::Size2M);
            return Err(AxError::InvalidInput);
        };

        for fault_addr in fault_addrs {
            // Fault in missing 4K pages so we can copy current contents into the new 2M page.
            // Use read-only access_flags; MAP_PRIVATE writes should go through COW later.
            match self.populate(
                VirtAddrRange::from_start_size(fault_addr, PAGE_SIZE_4K),
                flags,
                flags - MappingFlags::WRITE,
                pt,
            ) {
                Ok((_, _)) => {
                    let (old_pa, _, _) = pt.query(fault_addr)?;
                    let offset = fault_addr - range.start;
                    let src = phys_to_virt(old_pa);
                    let dst_pa = new_pa + offset;
                    let dst = phys_to_virt(dst_pa);
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            src.as_ptr(),
                            dst.as_mut_ptr(),
                            PAGE_SIZE_4K,
                        );
                    }
                    old_pages.push(old_pa);
                }
                Err(_) => return Err(AxError::BadAddress),
            }
        }

        if pt.remap_huge(range.start, new_pa, flags, PageSize::Size2M).is_err() {
            dealloc_frame(new_pa, PageSize::Size2M);
            return Err(AxError::BadAddress);
        }

        for offset in (0..THP_PAGE_BYTES).step_by(PAGE_SIZE_4K) {
            inc_frame_ref(new_pa + offset);
        }

        for frame in old_pages {
            if dec_frame_ref(frame) == 1 {
                dealloc_frame(frame, PageSize::Size4K);
            }
        }
        Ok(())
    }

    fn try_collapse_page(
        &self,
        range: VirtAddrRange,
        pt: &mut PageTableMut,
        max_ptes_none: usize,
        max_ptes_shared: usize,
        pages_scanned: &mut usize,
        pages_to_scan: usize,
    ) -> AxResult<bool> {
        let mut pte_none = 0;
        let mut shared_ptes = 0;

        for page_va in pages_in(range, PageSize::Size4K)? {
            if *pages_scanned >= pages_to_scan {
                return Ok(false);
            }
            *pages_scanned += 1;

            match pt.query(page_va) {
                Ok((paddr, _flags, _)) => {
                    if frame_ref_count(paddr) > 1 {
                        shared_ptes += 1;
                        if shared_ptes > max_ptes_shared {
                            return Ok(false);
                        }
                    }
                }
                Err(PagingError::NotMapped) => {
                    pte_none += 1;
                    if pte_none > max_ptes_none {
                        return Ok(false)
                    }
                }
                Err(_) => return Ok(false),
            }
        }

        self.collapse_page(range, pt, false, None)?;
        Ok(true)
    }

    fn set_vma_flag(&self, flag: VmaFlags) {
        let mut v = self.vma_flags.lock();
        *v |= flag;
    }

    fn clear_vma_flag(&self, flag: VmaFlags) {
        let mut v = self.vma_flags.lock();
        v.remove(flag);
    }

    fn contain_vma_flag(&self, flag: VmaFlags) -> bool {
        self.vma_flags.lock().contains(flag)
    }

    fn transparent_hugepage_enabled(&self) -> bool {
        let v = *self.vma_flags.lock();
        if v.contains(VmaFlags::VM_NOHUGEPAGE) {
            return false;
        }

        match *GLOBAL_THP_POLICY.lock() {
            ThpPolicy::Always => true,
            ThpPolicy::Madvise => v.contains(VmaFlags::VM_HUGEPAGE),
            ThpPolicy::Never => false,
        }
    }

    fn page_size(&self) -> PageSize {
        self.size
    }

    fn map(&self, range: VirtAddrRange, flags: MappingFlags, _pt: &mut PageTableMut) -> AxResult {
        debug!("Cow::map: {range:?} {flags:?}",);
        Ok(())
    }

    fn unmap(&self, range: VirtAddrRange, pt: &mut PageTableMut) -> AxResult {
        debug!("Cow::unmap: {range:?}");
        if !(self.size as usize).is_power_of_two()
            || !range.start.is_aligned(self.size)
            || !range.end.is_aligned(self.size)
        {
            return Err(AxError::InvalidInput);
        }

        let mut va = range.start;
        let end = range.end;
        let backend_page_size = self.size as usize;
        while va < end {
            match pt.query(va) {
                Ok((paddr, _, page_size)) => {
                    if page_size == PageSize::Size2M 
                    && backend_page_size == PAGE_SIZE_4K {
                        let va_usize: usize = va.into();
                        let end_usize: usize = end.into();
                        let huge_start = va_usize & !(THP_PAGE_BYTES - 1);
                        let huge_end = huge_start + THP_PAGE_BYTES;
                        // If the page is a THP and start address
                        // is aligned, the range contain the 
                        // THP range, unmap the whole THP 
                        // all at one at 2M granularity
                        if va_usize == huge_start && huge_end <= end_usize {
                            let base_va: VirtAddr = huge_start.into();
                            let (base_pa, _, _) = pt.query(base_va)?;

                            for offset in (0..THP_PAGE_BYTES).step_by(PAGE_SIZE_4K) {
                                if dec_frame_ref(base_pa + offset) == 1 {
                                    dealloc_frame(base_pa + offset, PageSize::Size4K);
                                }
                            }

                            pt.unmap(base_va)?;
                            va = (base_va + THP_PAGE_BYTES).into();
                            continue;
                        }

                        // Otherwise split the THP in the old page table
                        // and retry at 4K granularity.
                        pt.split_huge_pmd(va)?;
                        continue;
                    } else {
                        if dec_frame_ref(paddr) == 1 {
                            dealloc_frame(paddr, page_size);
                        }
                        pt.unmap(va)?;
                        let step: usize = page_size.into();
                        va += step;
                    }
                }
                Err(_) => {
                    // Deallocation is needn't if the page is not allocated
                    va += backend_page_size;
                }
            }
        }
        Ok(())
    }

    fn populate(
        &self,
        range: VirtAddrRange,
        flags: MappingFlags,
        access_flags: MappingFlags,
        pt: &mut PageTableMut,
        ) -> AxResult<(usize, Option<Box<dyn FnOnce(&mut AddrSpace)>>)> {
        if !(self.size as usize).is_power_of_two()
            || !range.start.is_aligned(self.size)
            || !range.end.is_aligned(self.size) 
        {
            return Err(AxError::InvalidInput);
        }

        let mut pages = 0;
        let mut va = range.start;
        let end = range.end;
        let backend_page_size: usize = self.size as usize;

        while va < end {
            match pt.query(va) {
                Ok((paddr, page_flags, page_size)) => {
                    if page_size == PageSize::Size2M && backend_page_size == PAGE_SIZE_4K {
                        pt.split_huge_pmd(va)?;
                        continue;
                    }
                    if access_flags.contains(MappingFlags::WRITE)
                    && !page_flags.contains(MappingFlags::WRITE)
                    {
                        self.handle_cow_fault(va, paddr, flags, pt)?;
                        pages += 1;
                    } else if page_flags.contains(access_flags) {
                        pages += 1;
                    } 
                    va += backend_page_size;
                }
                // If the page is not mapped, try map it.
                Err(PagingError::NotMapped) => {
                    // Allocate one backend-sized page starting at `va`.
                    self.alloc_new_at(va, flags, pt)?;
                    pages += 1;
                    va += backend_page_size;
                }
                Err(_) => return Err(AxError::BadAddress),
            }
        }

        Ok((pages, None))
    }

fn clone_map(
        &self,
        range: VirtAddrRange,
        flags: MappingFlags,
        old_pt: &mut PageTableMut,
        new_pt: &mut PageTableMut,
        _new_aspace: &Arc<Mutex<AddrSpace>>,
    ) -> AxResult<Backend> {
        if !(self.size as usize).is_power_of_two()
            || !range.start.is_aligned(self.size)
            || !range.end.is_aligned(self.size)
        {
            return Err(AxError::InvalidInput);
        }

        let cow_flags = flags - MappingFlags::WRITE;
        let mut va = range.start;
        let end = range.end;
        let backend_page_size = self.size as usize;
        while va < end {
            match old_pt.query(va) {
                Ok((paddr, _old_flags, page_size)) => {
                    if page_size == PageSize::Size2M  && 
                    backend_page_size == PAGE_SIZE_4K {
                        let va_usize: usize = va.into();
                        let end_usize: usize = end.into();
                        let huge_start = va_usize & !(THP_PAGE_BYTES - 1);
                        let huge_end = huge_start + THP_PAGE_BYTES;

                        if va_usize == huge_start && huge_end <= end_usize {
                            let base_va: VirtAddr = huge_start.into();
                            let (base_pa, _, _) = old_pt.query(base_va)?;

                            for offset in (0..THP_PAGE_BYTES).step_by(PAGE_SIZE_4K) {
                                inc_frame_ref(base_pa + offset);
                            }

                            old_pt.protect(base_va, cow_flags)?;
                            new_pt.map(base_va, base_pa, PageSize::Size2M, cow_flags)?;
                            va = (base_va + THP_PAGE_BYTES).into();
                            continue;
                        }
                        old_pt.split_huge_pmd(va)?;
                        continue;
                    } else {
                        inc_frame_ref(paddr);
                        old_pt.protect(va, cow_flags)?;
                        new_pt.map(va, paddr, page_size, cow_flags)?;
                        let step: usize = page_size.into();
                        va += step;
                    }
                }
                // If the page is not mapped, skip it.
                Err(PagingError::NotMapped) => {
                    va += backend_page_size;
                }
                Err(_) => return Err(AxError::BadAddress),
            }
        }

        Ok(Backend::Cow(self.clone()))
    }
}

impl Backend {
    pub fn new_cow(
        start: VirtAddr,
        size: PageSize,
        file: FileBackend,
        file_start: u64,
        file_end: Option<u64>,
        flags: VmaFlags
    ) -> Self {
        Self::Cow(CowBackend {
            start,
            size,
            file: Some((file, file_start, file_end)),
            vma_flags: Mutex::new(flags),
        })
    }

    pub fn new_alloc(start: VirtAddr, size: PageSize, flags: VmaFlags) -> Self {
        Self::Cow(CowBackend {
            start,
            size,
            file: None,
            vma_flags: Mutex::new(flags),
        })
    }
}

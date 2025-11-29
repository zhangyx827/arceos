use alloc::{boxed::Box, collections::btree_map::BTreeMap, string::{String, ToString}, sync::Arc, vec::Vec};
use axfs_ng_vfs::{VfsError, VfsResult};
use core::slice;

use axerrno::{AxError, AxResult};
use axfs_ng::FileBackend;
use axhal::{
    mem::phys_to_virt,
    paging::{MappingFlags, PageSize, PageTableMut, PagingError},
};
use axsync::Mutex;
use kspin::SpinNoIrq;
use memory_addr::{PAGE_SIZE_4K, PAGE_SIZE_2M, PageIter4K, PhysAddr, VirtAddr, VirtAddrRange};
use lazy_static::lazy_static;
use crate::{
    AddrSpace,
    backend::{Backend, BackendOps, VmaFlags, alloc_frame, dealloc_frame, pages_in},
};


static FRAME_TABLE: SpinNoIrq<BTreeMap<PhysAddr, u8>> = SpinNoIrq::new(BTreeMap::new());

// The global thp policy controled through 
// /sys/kernel/mm/tranparent_hugepage/enabled
lazy_static! {
    static ref GLOBAL_THP_POLICY: SpinNoIrq<String> = SpinNoIrq::new(String::from("madvise\n"));
}

pub fn modify_ano_policy(policy: &str) -> VfsResult<Vec<u8>> {
    if !policy.eq("never")
    && !policy.eq("madvise")
    && !policy.eq("always") 
    && !policy.eq("") {
        return Err(VfsError::InvalidInput);
    }
    *GLOBAL_THP_POLICY.lock() = policy.to_string() + "\n";
    Ok(Vec::new())
}

pub fn current_ano_policy() -> String {
    let _s = (*GLOBAL_THP_POLICY.lock()).clone().to_string();
    (*GLOBAL_THP_POLICY.lock().clone()).to_string()
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

pub struct VmaFlagsWrapper(Mutex<VmaFlags>);

/// Copy-on-write mapping backend.
///
/// This corresponds to the `MAP_PRIVATE` flag.
#[derive(Clone)]
pub struct CowBackend {
    start: VirtAddr,
    size: PageSize,
    file: Option<(FileBackend, u64, Option<u64>)>,
    vma_flags: Arc<VmaFlagsWrapper>
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
    fn collapse_page(
        &self,
        m_start: VirtAddr,
        m_end: VirtAddr,
        pt: &mut PageTableMut,
    ) -> AxResult {
        let new_pa = alloc_frame(true, PageSize::Size2M)?;

        let mut flags_opt: Option<MappingFlags> = None;
        let mut old_pages = Vec::new();
        // Copy content from existing mappings into the new 2M page.
        for page_va in PageIter4K::new(m_start, m_end).expect("4KB aligned range") {
            let offset = page_va - m_start;
            if let Ok((old_pa, page_flags, _)) = pt.query(page_va) {
                if flags_opt.is_none() {
                    flags_opt = Some(page_flags);
                }
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
            // holes remain zero
        }

        let Some(flags) = flags_opt else {
            // No mapped 4K pages; free the newly allocated huge page.
            dealloc_frame(new_pa, PageSize::Size2M);
            return Ok(());
        };

        if pt.remap_huge(m_start, new_pa, flags, PageSize::Size2M).is_err() {
            dealloc_frame(new_pa, PageSize::Size2M);
        }

        for offset in (0..PAGE_SIZE_2M).step_by(PAGE_SIZE_4K) {
            inc_frame_ref(new_pa + offset);
        }

        for frame in old_pages {
            if dec_frame_ref(frame) == 1 {
                dealloc_frame(frame, PageSize::Size4K);
            }
        }
        Ok(())
    }

    fn set_vma_flag(&self, vma_flags: VmaFlags) {
        *self.vma_flags.0.lock() |= vma_flags;
    }

    fn clear_vma_flag(&self, vma_flags: VmaFlags) {
        *self.vma_flags.0.lock() ^= vma_flags;
    }

    fn transparent_hugepage_enabled(&self) -> bool {
        return *self.vma_flags.0.lock() == VmaFlags::HUGEPAGE
        || current_ano_policy().eq("always\n")
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
        let mut va = range.start;
        let end = range.end;
        let backend_page_size = self.size as usize;
        while va < end {
            match pt.query(va) {
                Ok((paddr, _old_flags, page_size)) => {
                    if page_size == PageSize::Size2M 
                    && backend_page_size == PAGE_SIZE_4K {
                        let va_usize: usize = va.into();
                        let end_usize: usize = end.into();
                        let huge_start = va_usize & !(PAGE_SIZE_2M - 1);
                        let huge_end = huge_start + PAGE_SIZE_2M;

                        if va_usize == huge_start && huge_end <= end_usize {
                            let base_va: VirtAddr = huge_start.into();
                            let (base_pa, _, _) = pt.query(base_va)?;

                            for offset in (0..PAGE_SIZE_2M).step_by(PAGE_SIZE_4K) {
                                dec_frame_ref(base_pa + offset);
                            }

                            pt.unmap(base_va)?;
                            va = (base_va + PAGE_SIZE_2M).into();
                            continue;
                        }

                        // Otherwise split the THP in the old page table
                        // and retry at 4K granularity.
                        pt.split_huge_pmd(va)?;
                        continue;
                    } else {
                        dec_frame_ref(paddr);
                        pt.unmap(va)?;
                        let step: usize = page_size.into();
                        va += step;
                    }
                }
                Err(_) => {
                    va += backend_page_size;
                }
            }
        }

        for addr in pages_in(range, self.size)? {
            if let Ok((frame, _flags, page_size)) = pt.unmap(addr) {
                // With khugepaged
                // Page size may not equal to `self.size`
                if dec_frame_ref(frame) == 1 {
                    dealloc_frame(frame, page_size);
                }
            } else {
                // Deallocation is needn't if the page is not allocated.
            }
        }
        Ok(())
    }

    fn pte_fault_collapse(
        &self,
        vaddr: VirtAddr,
        flags: MappingFlags,
        _access_flags: MappingFlags,
        pt: &mut PageTableMut,
    ) -> AxResult<(usize, Option<Box<dyn FnOnce(&mut AddrSpace)>>)> {
        let m_range = VirtAddrRange::from_start_size(vaddr, PageSize::Size2M as _);
        for addr in pages_in(m_range, PageSize::Size4K)? {
            if pt.query(addr).is_ok() {
                return Err(PagingError::NotMapped.into());
            }
        }

        let frame = alloc_frame(true, PageSize::Size2M)?;

        pt.map(vaddr, frame, PageSize::Size2M, flags)?;
        return Ok((1, None))
    }

    fn populate(
        &self,
        range: VirtAddrRange,
        flags: MappingFlags,
        access_flags: MappingFlags,
        pt: &mut PageTableMut,
        ) -> AxResult<(usize, Option<Box<dyn FnOnce(&mut AddrSpace)>>)> {
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
                        let huge_start = va_usize & !(PAGE_SIZE_2M - 1);
                        let huge_end = huge_start + PAGE_SIZE_2M;

                        // If the clone range fully covers this THP and we
                        // are at its base, keep it as a huge mapping and
                        // bump refs for all 4K pages underneath.
                        if va_usize == huge_start && huge_end <= end_usize {
                            let base_va: VirtAddr = huge_start.into();
                            let (base_pa, _f, _s) = old_pt.query(base_va)?;

                            for offset in (0..PAGE_SIZE_2M).step_by(PAGE_SIZE_4K) {
                                inc_frame_ref(base_pa + offset);
                            }

                            old_pt.protect(base_va, cow_flags)?;
                            new_pt.map(base_va, base_pa, PageSize::Size2M, cow_flags)?;
                            va = (base_va + PAGE_SIZE_2M).into();
                            continue;
                        }

                        // Otherwise split the THP in the old page table
                        // and retry at 4K granularity.
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
    ) -> Self {
        Self::Cow(CowBackend {
            start,
            size,
            file: Some((file, file_start, file_end)),
            vma_flags: Arc::new(VmaFlagsWrapper(VmaFlags::empty().into())),
        })
    }

    pub fn new_alloc(start: VirtAddr, size: PageSize) -> Self {
        Self::Cow(CowBackend {
            start,
            size,
            file: None,
            vma_flags: Arc::new(VmaFlagsWrapper(VmaFlags::empty().into()))
        })
    }
}

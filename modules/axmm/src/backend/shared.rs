use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};

use axerrno::{AxError, AxResult};
use axfs_ng::{CachedFile, PageOperation};
use axfs_ng_vfs::{VfsError, VfsResult};
use axhal::paging::{MappingFlags, PageSize, PageTableMut};
use axsync::Mutex;
use kspin::SpinNoIrq;
use lazy_static::lazy_static;
use memory_addr::{MemoryAddr, PAGE_SIZE_4K, VirtAddr, VirtAddrRange};
use page_table_multiarch::PagingError;

use crate::{
    AddrSpace,
    THP_PAGE_BYTES,
    backend::{
        Backend, BackendOps, VmaFlags, pages_in, register_cache_listener,
    },
};

// Global THP policy for shmem/tmpfs controlled via
// /sys/kernel/mm/transparent_hugepage/shmem_enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShmemThpPolicy {
    Always,
    Advise,
    Never,
}

impl ShmemThpPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Always => "always",
            Self::Advise => "advise",
            Self::Never => "never",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "always" => Some(Self::Always),
            "advise" => Some(Self::Advise),
            "never" => Some(Self::Never),
            _ => None,
        }
    }
}

lazy_static! {
    static ref SHMEM_THP_POLICY: SpinNoIrq<ShmemThpPolicy> = SpinNoIrq::new(ShmemThpPolicy::Advise);
}

/// Updates the shmem/tmpfs THP policy string (e.g. "always", "advise",
/// "never").
pub fn set_shmem_thp_policy(policy: &str) -> VfsResult<Vec<u8>> {
    // Some writers may perform a truncate-like write with an empty buffer before
    // writing the real contents. Treat empty writes as a no-op for robustness.
    let trimmed = policy.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let Some(policy) = ShmemThpPolicy::parse(trimmed) else {
        return Err(VfsError::InvalidInput);
    };
    *SHMEM_THP_POLICY.lock() = policy;
    Ok(Vec::new())
}

/// Returns the current shmem/tmpfs THP policy string 
pub fn current_shmem_thp_policy() -> String {
    SHMEM_THP_POLICY.lock().as_str().into()
}

pub struct SharedBackendInner {
    start: VirtAddr,
    cache: Arc<CachedFile>,
    // Per-VMA THP policy flags for shmem/tmpfs mappings.
    vma_flags: Mutex<VmaFlags>,
}

impl SharedBackendInner {
    pub fn register_listener(self: &Arc<Self>, aspace: &Arc<Mutex<AddrSpace>>) -> usize {
        register_cache_listener(&self.cache, &self, aspace, |backend, pn, aspace, op| {
            backend.on_evict(pn, aspace, op)
        })
    }

    fn on_evict(self: &Arc<Self>, pn: u32, aspace: &mut AddrSpace, op: PageOperation) {
        let vaddr = self.start + pn as usize * PageSize::Size4K as usize;
        super::on_evict(self, vaddr, aspace, op);
    }
}

#[derive(Clone)]
pub struct SharedBackend(Arc<SharedBackendInner>);
impl SharedBackend {
    pub fn cache(&self) -> &Arc<CachedFile> {
        &self.0.cache
    }

    pub(super) fn ptr_eq_inner(&self, inner: &SharedBackendInner) -> bool {
        core::ptr::eq(&*self.0, inner)
    }
}

impl BackendOps for SharedBackend {
    fn try_collapse_page(
        &self,
        range: VirtAddrRange,
        pt: &mut PageTableMut,
        max_ptes_none: usize,
        _max_ptes_shared: usize,
        pages_scanned: &mut usize,
        pages_to_scan: usize,
    ) -> AxResult<bool> {
        let mut pte_none = 0;

        for (_, addr) in pages_in(range, PageSize::Size4K)?.enumerate() {
            if *pages_scanned >= pages_to_scan {
                return Ok(false);
            }
            *pages_scanned += 1;

            match pt.query(addr) {
                Ok((..)) => {}
                Err(PagingError::NotMapped) => {
                    pte_none += 1;
                    if pte_none > max_ptes_none {
                        return Ok(false);
                    }
                }
                Err(_) => return Ok(false),
            }
        }

        self.collapse_page(range, pt, false, None)?;
        Ok(true)
    }

    fn collapse_page(
        &self,
        range: VirtAddrRange,
        pt: &mut PageTableMut,
        fault_in: bool,
        _callbacks: Option<&mut Vec<Box<dyn FnOnce(&mut AddrSpace)>>>,
    ) -> AxResult {
        let start_pn = ((range.start - self.0.start) / PAGE_SIZE_4K) as u32;
        let mut flags_opt: Option<MappingFlags> = None;
        for (_, addr) in pages_in(range, PageSize::Size4K)?.enumerate() {
            match pt.query(addr) {
                Ok((_, page_flags, _)) => {
                    if flags_opt.is_none() {
                        flags_opt = Some(page_flags);
                    }
                }
                Err(PagingError::NotMapped) => {}
                _ => {
                    return Err(AxError::BadAddress);
                }
            }
        }

        let Some(flags) = flags_opt else {
            // At least one page must currently be backed by physical memory.
            return Ok(());
        };

        let new_pa = self.0.cache.replace_with_huge_page(start_pn)?;
        pt.remap_huge(range.start, new_pa, flags, PageSize::Size2M)?;
        Ok(())
    }

    fn demote_huge(&self, range: VirtAddrRange, pt: &mut PageTableMut) -> AxResult {
        // If a 2MiB PMD mapping is only partially covered by this range, demote it
        // to 4KiB PTEs and also split the underlying page cache huge chunk so that
        // future 4KiB operations remain consistent.
        let range_end = range.end;
        let mut vaddr = range.start.align_down(PageSize::Size2M);
        while vaddr < range_end {
            let huge_start = vaddr;
            let huge_end = huge_start + THP_PAGE_BYTES;
            if range.start > huge_start || range_end < huge_end {
                if let Ok((_, _, page_size)) = pt.query(huge_start) {
                    if page_size == PageSize::Size2M {
                        pt.split_huge_pmd(huge_start)?;

                        let offset = huge_start - self.0.start;
                        let pn = offset / PAGE_SIZE_4K;
                        let (base_opt, chunk_size) = self.0.cache.locate_chunk(pn as u32);
                        if base_opt.is_some() && chunk_size == THP_PAGE_BYTES {
                            self.0.cache.split_huge_page(pn as u32)?;
                        }
                    }
                }
            }
            vaddr += THP_PAGE_BYTES;
        }
        Ok(())
    }

    fn set_vma_flag(&self, flag: VmaFlags) {
        let mut v = self.0.vma_flags.lock();
        *v |= flag;
    }

    fn clear_vma_flag(&self, flag: VmaFlags) {
        let mut v = self.0.vma_flags.lock();
        v.remove(flag);
    }

    fn contain_vma_flag(&self, flag: VmaFlags) -> bool {
        self.0.vma_flags.lock().contains(flag)
    }

    fn transparent_hugepage_enabled(&self) -> bool {
        let v = *self.0.vma_flags.lock();
        if v.contains(VmaFlags::VM_NOHUGEPAGE) {
            return false;
        }
        
        match *SHMEM_THP_POLICY.lock() {
            ShmemThpPolicy::Always => true,
            ShmemThpPolicy::Advise => v.contains(VmaFlags::VM_HUGEPAGE),
            ShmemThpPolicy::Never => false,
        }
    }

    fn map(&self, _range: VirtAddrRange, _flags: MappingFlags, _pt: &mut PageTableMut) -> AxResult {
        debug!("Shared::map: {:?} {:?}", _range, _flags);
        Ok(())
    }

    fn unmap(&self, range: VirtAddrRange, pt: &mut PageTableMut) -> AxResult {
        debug!("Shared::unmap: {:?}", range);
        if !range.start.is_aligned(PAGE_SIZE_4K) || !range.end.is_aligned(PAGE_SIZE_4K) {
            return Err(AxError::InvalidInput);
        }

        let mut va = range.start;
        let end = range.end;
        while va < end {
            match pt.query(va) {
                Ok((_, _, page_size)) => {
                    if page_size == PageSize::Size2M {
                        let va_usize: usize = va.into();
                        let end_usize: usize = end.into();
                        let huge_start = va_usize & !(THP_PAGE_BYTES - 1);
                        let huge_end = huge_start + THP_PAGE_BYTES;
                        if va_usize == huge_start && huge_end <= end_usize {
                            let base_va: VirtAddr = huge_start.into();
                            let (base_pa, ..) = pt.query(base_va)?;
                            pt.unmap(base_va)?;
                            va = (base_va + THP_PAGE_BYTES).into();
                            continue;
                        }

                        pt.split_huge_pmd(va)?;
                        let offset = va - self.0.start;
                        let pn = offset / PAGE_SIZE_4K;
                        self.0.cache.split_huge_page(pn as u32)?;
                        continue;
                    } else {
                        pt.unmap(va)?;
                        let step: usize = page_size.into();
                        va += step;
                    }
                }
                Err(PagingError::NotMapped) => {
                    va += PAGE_SIZE_4K;
                }
                Err(err) => {
                    warn!("Failed to unmap page {:?}: {:?}", va, err);
                    return Err(err.into());
                }
            }
        }
        Ok(())
    }

    fn clone_map(
        &self,
        _range: VirtAddrRange,
        _flags: MappingFlags,
        _old_pt: &mut PageTableMut,
        _new_pt: &mut PageTableMut,
        _new_aspace: &Arc<Mutex<AddrSpace>>,
    ) -> AxResult<Backend> {
        Ok(Backend::Shared(self.clone()))
    }

    fn page_size(&self) -> PageSize {
        PageSize::Size4K
    }

    fn populate(
        &self,
        range: VirtAddrRange,
        flags: MappingFlags,
        access_flags: MappingFlags,
        pt: &mut PageTableMut,
    ) -> AxResult<(usize, Option<Box<dyn FnOnce(&mut AddrSpace)>>)> {
        let mut pages = 0;
        let start_page = ((range.start - self.0.start) / PAGE_SIZE_4K) as u32;
        let (opt_num, size) = self.0.cache.locate_chunk(start_page);
        let mut page_size = PAGE_SIZE_4K;
        let mut pn = start_page;
        let mut base_pn = pn;

        if let Some(chunk_num) = opt_num {
            base_pn = chunk_num;
            page_size = size;
        }

        let mut va = range.start;
        let end = range.end;
        while va < end {
            match pt.query(va) {
                Ok((paddr, page_flags, page_size)) => {
                    let base_va = va.align_down(page_size);
                    let off = va - base_va;
                    va += page_size as usize - off;
                    pn += (page_size as usize - off) as u32 / PAGE_SIZE_4K as u32;
                    pages += 1;
                }
                // If the page is not mapped, try map it.
                Err(PagingError::NotMapped) => {
                    self.0
                        .cache
                        .with_page_or_insert(base_pn, PAGE_SIZE_4K, |page, _| {
                            let page_size = page.size();
                            if va.is_aligned(page_size)
                                && range.contains(va)
                                && (range.end - va) >= page_size.into()
                            {
                                let size = if page_size == PAGE_SIZE_4K {
                                    PageSize::Size4K
                                } else {
                                    PageSize::Size2M
                                };
                                pt.map(va, page.paddr(), size, flags)?;
                                pages += 1;
                                va += page_size;
                                pn += page_size as u32 / PAGE_SIZE_4K as u32;
                                base_pn += page_size as u32 / PAGE_SIZE_4K as u32;
                                return Ok(());
                            }
                            // Fall back to 4KiB
                            // `base_pn` is not changed here
                            let pa = page.paddr() + (pn - base_pn) as usize * PAGE_SIZE_4K;
                            pt.map(va, pa, PageSize::Size4K, flags)?;
                            pages += 1;
                            va += PAGE_SIZE_4K;
                            pn += 1;
                            Ok(())
                        })?;
                }
                Err(_) => return Err(AxError::BadAddress),
            }
        }
        Ok((pages, None))
    }
}

impl Backend {
    pub fn new_shared(
        start: VirtAddr,
        cache: Arc<CachedFile>,
        aspace: &Arc<Mutex<AddrSpace>>,
    ) -> Self {
        let inner = Arc::new(SharedBackendInner {
            start,
            cache,
            vma_flags: Mutex::new(VmaFlags::empty()),
        });
        inner.register_listener(aspace);
        Self::Shared(SharedBackend(inner))
    }
}

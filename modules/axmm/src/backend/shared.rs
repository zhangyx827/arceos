use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};
use axfs_ng::CachedFile;
use page_table_multiarch::PagingError;

use axerrno::{AxError, AxResult};
use axfs_ng_vfs::{VfsError, VfsResult};
use axhal::paging::{MappingFlags, PageSize, PageTableMut};
use axsync::Mutex;
use kspin::SpinNoIrq;
use lazy_static::lazy_static;
use memory_addr::{MemoryAddr, PAGE_SIZE_2M, PAGE_SIZE_4K, VirtAddr, VirtAddrRange};

use super::{alloc_frame, dealloc_frame};
use crate::{
    AddrSpace, backend::{Backend, BackendOps, VmaFlags, divide_page, on_evict, pages_in, register_cache_listener}
};

// Global THP policy for shmem/tmpfs controlled via
// /sys/kernel/mm/transparent_hugepage/shmem_enabled.
lazy_static! {
    static ref SHMEM_THP_POLICY: SpinNoIrq<String> =
        SpinNoIrq::new(String::from("madvise"));
}

/// Updates the shmem/tmpfs THP policy string (e.g. "always", "madvise", "never").
pub fn set_shmem_thp_policy(policy: &str) -> VfsResult<Vec<u8>> {
    if policy != "never" && policy != "madvise" && policy != "always" && !policy.is_empty() {
        return Err(VfsError::InvalidInput);
    }
    *SHMEM_THP_POLICY.lock() = alloc::format!("{policy}");
    Ok(Vec::new())
}

/// Returns the current shmem/tmpfs THP policy string (including trailing '\n').
pub fn current_shmem_thp_policy() -> String {
    SHMEM_THP_POLICY.lock().clone()
}

pub struct SharedBackendInner {
    start: VirtAddr,
    cache: Arc<CachedFile>,
    // Per-VMA THP policy flags for shmem/tmpfs mappings.
    vma_flags: Mutex<VmaFlags>,
}

impl SharedBackendInner {
    pub fn register_listener(self: &Arc<Self>, aspace: &Arc<Mutex<AddrSpace>>) -> usize {
        register_cache_listener(&self.cache, &self, aspace, |backend, pn, aspace| {
            backend.on_evict(pn, aspace)
        })
    }

    fn on_evict(self: &Arc<Self>, pn: u32, aspace: &mut AddrSpace) {
        let vaddr = self.start + pn as usize * PageSize::Size4K as usize;
        super::on_evict(self, vaddr, aspace);
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
                Ok((_, _, _)) => {
                }
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
        range:VirtAddrRange,
        pt: &mut PageTableMut,
        _fault_in:bool,
        _callbacks: Option<&mut Vec<Box<dyn FnOnce(&mut AddrSpace)>>>,
    ) -> AxResult {
        let start_pn = (range.start.as_usize() / PAGE_SIZE_4K) as u32;
        let mut flags_opt: Option<MappingFlags> = None;
        for (_, addr) in pages_in(range, PageSize::Size4K)?.enumerate() {
            match pt.query(addr) {
                Ok((_, page_flags, _)) => {
                    if flags_opt.is_none() {
                        flags_opt = Some(page_flags);
                    }
                },
                Err(PagingError::NotMapped) => {
                }
                _ => {
                    return Err(AxError::BadAddress);
                },
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

    fn set_vma_flag(&self, flag: VmaFlags) {
        if current_shmem_thp_policy().eq("advise") {
            let mut v = self.0.vma_flags.lock();
            *v |= flag;
        }
    }

    fn clear_vma_flag(&self, flag: VmaFlags) {
        if current_shmem_thp_policy().eq("always") {
            let mut v = self.0.vma_flags.lock();
            *v ^= flag;
        }
    }

    fn contain_vma_flag(&self, flag: VmaFlags) -> bool {
        self.0.vma_flags.lock().contains(flag)
    }


    fn transparent_hugepage_enabled(&self) -> bool {
        let v = *self.0.vma_flags.lock();
        v.contains(VmaFlags::VM_HUGEPAGE)
            || current_shmem_thp_policy().eq("always\n")
    }

    fn map(&self, _range: VirtAddrRange, _flags: MappingFlags, _pt: &mut PageTableMut) -> AxResult {
        debug!("Shared::map: {:?} {:?}", _range, _flags);
        Ok(())
    }

    fn unmap(&self, range: VirtAddrRange, pt: &mut PageTableMut) -> AxResult {
        debug!("Shared::unmap: {:?}", range);
        if !range.start.is_aligned(PAGE_SIZE_4K)
            || !range.end.is_aligned(PAGE_SIZE_4K)
        {
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
                        let huge_start = va_usize & !(PAGE_SIZE_2M - 1);
                        let huge_end = huge_start + PAGE_SIZE_2M;
                        if va_usize == huge_start && huge_end <= end_usize {
                            let base_va: VirtAddr = huge_start.into();
                            let (base_pa, _, _) = pt.query(base_va)?;
                            pt.unmap(base_va)?;
                            va = (base_va + PAGE_SIZE_2M).into();
                            continue;
                        }

                        pt.split_huge_pmd(va)?;
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
    ) -> AxResult<(usize,Option<Box<dyn FnOnce(&mut AddrSpace)>>)> {
        let mut pages = 0;
        let start_page = range.start.as_usize() as u32 / PAGE_SIZE_4K as u32;
        for (i, addr) in pages_in(range, PageSize::Size4K)?.enumerate() {
            let pn = start_page + i as u32;
            match pt.query(addr) {
                Ok((paddr, page_flags, _)) => {
                    pages += 1;
                }
                Err(PagingError::NotMapped) => {
                    let mut page_size = PAGE_SIZE_4K;
                    let (opt_num, size) = self.0.cache.locate_chunk(pn, pn);
                    let mut base_pn = pn;
                    if let Some(chunk_num) = opt_num {
                        base_pn = chunk_num;
                        page_size = size
                    }
                    self.0.cache.with_page_or_insert(base_pn, page_size, |page, _| {
                        // No need to evict cache
                        let pa = page.paddr() + (pn - base_pn) as usize * PAGE_SIZE_4K;
                        pt.map(addr, pa, PageSize::Size4K, flags)?;
                        pages += 1;
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
        aspace: &Arc<Mutex<AddrSpace>>
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

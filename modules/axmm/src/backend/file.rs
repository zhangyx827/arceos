use alloc::{
    boxed::Box,
    sync::{Arc, Weak},
    vec::Vec,
};
use core::sync::atomic::{AtomicUsize, Ordering};

use axerrno::{AxError, AxResult};
use axfs_ng::{CachedFile, FileFlags};
use axhal::{
    paging::{MappingFlags, PageSize, PageTableMut, PagingError},
};
use axsync::{Mutex, MutexGuard};
use memory_addr::{MemoryAddr, PAGE_SIZE_2M, PAGE_SIZE_4K, VirtAddr, VirtAddrRange};

use crate::{
    AddrSpace, backend::{Backend, BackendOps, VmaFlags, current_shmem_thp_policy, pages_in, 
        register_cache_listener, on_evict}
};

#[doc(hidden)]
pub struct FileBackendInner {
    start: VirtAddr,
    cache: CachedFile,
    flags: FileFlags,
    offset_page: u32,
    handle: AtomicUsize,
    futex_handle: Arc<()>,
    // Per-VMA THP policy for this file mapping (typically tmpfs/shmem).
    vma_flags: Mutex<VmaFlags>,
}

impl Drop for FileBackendInner {
    fn drop(&mut self) {
        let handle = self.handle.load(Ordering::Acquire);
        if handle != 0 {
            unsafe {
                self.cache.remove_evict_listener(handle);
            }
        }
    }
}

impl FileBackendInner {
    pub fn register_listener(self: &Arc<Self>, aspace: &Arc<Mutex<AddrSpace>>) -> usize {
        register_cache_listener(&self.cache, &self, aspace, |backend, pn, aspace| {
            backend.on_evict(pn, aspace)
        })
    }

    fn on_evict(
        self: &Arc<Self>, 
        pn: u32,
        aspace: &mut AddrSpace
    ) {
        let Some(pn) = pn.checked_sub(self.offset_page) else {
            return;
        };  
        let vaddr = self.start + pn as usize * PageSize::Size4K as usize;
        super::on_evict(self, vaddr, aspace);
    }

    pub fn transparent_hugepage_enabled(&self) -> bool {
        // Only support tmpfs 
        if !self.cache.in_memory() {
            return false;
        }
        let v = *self.vma_flags.lock();
        v.contains(VmaFlags::VM_HUGEPAGE)
        || current_shmem_thp_policy().eq("always\n")
    }

    /// Sets per-VMA THP-related flags (e.g. VM_HUGEPAGE / VM_NOHUGEPAGE).
    pub fn set_vma_flag(&self, flag: VmaFlags) {
        // Follow the same policy as shmem: only honor explicit per-VMA hints
        // when the global shmem policy is "advise".
        if current_shmem_thp_policy().eq("advise") {
            let mut v = self.vma_flags.lock();
            *v |= flag;
        }   
    }

    /// Clears per-VMA THP-related flags.
    pub fn clear_vma_flag(&self, flag: VmaFlags) {
    // When shmem policy is "always", per-VMA "nohuge" overrides are relevant.
        if current_shmem_thp_policy().eq("always") {
            let mut v = self.vma_flags.lock();
            *v ^= flag;
        }
    }
}

/// File-backed mapping backend.
#[derive(Clone)]
pub struct FileBackend(Arc<FileBackendInner>);
impl FileBackend {
    pub fn is_in_memory(&self) -> bool {
        self.0.cache.in_memory()
    }

    fn check_flags(&self, flags: MappingFlags) -> AxResult {
        let mut required_flags = FileFlags::empty();
        if flags.contains(MappingFlags::READ) {
            required_flags |= FileFlags::READ;
        }
        if flags.contains(MappingFlags::WRITE) {
            required_flags |= FileFlags::WRITE;
        }

        if !self.0.flags.contains(required_flags) {
            return Err(AxError::PermissionDenied);
        }
        Ok(())
    }

    pub fn futex_handle(&self) -> Weak<()> {
        Arc::downgrade(&self.0.futex_handle)
    }

    pub(super) fn ptr_eq_inner(&self, inner: &FileBackendInner) -> bool {
        core::ptr::eq(&*self.0, inner)
    }
}

impl BackendOps for FileBackend {
    fn collapse_page(
        &self,
        // 2MiB range
        range: VirtAddrRange,
        pt: &mut PageTableMut,
        fault_in: bool,
        mut callbacks: Option<&mut Vec<Box<dyn FnOnce(&mut AddrSpace)>>>,
    ) -> AxResult {
        let start_pn = ((range.start - self.0.start) / PAGE_SIZE_4K) as u32 + self.0.offset_page;
        let end_pn = start_pn + (range.size() / PAGE_SIZE_4K) as u32;
        let mut has_partial_huge = false;
        let mut start_chunk = 0;
        let mut pn = start_pn;

        while pn < end_pn {
            let (chunk_pn_opt, chunk_size) = self.0.cache.locate_chunk(start_chunk, pn);
            if let Some(chunk_pn) = chunk_pn_opt {
                if chunk_size == PAGE_SIZE_2M {
                    if chunk_pn > start_pn || chunk_pn + 512 < end_pn {
                        return Ok(())
                    }
                }
            }
            pn += 1;
            start_chunk = pn;
        }
        
        let mut flags_opt: Option<MappingFlags> = None;
        let mut fault_addrs = Vec::new();
        for (_, addr) in pages_in(range, PageSize::Size4K)?.enumerate() {
            match pt.query(addr) {
                Ok((_, page_flags, _)) => {
                    if flags_opt.is_none() {
                        flags_opt = Some(page_flags);
                    }
                },
                Err(PagingError::NotMapped) => {
                    if fault_in {
                        fault_addrs.push(addr);
                    }
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

        let reused_huge = self.0.cache.with_page(start_pn, |page| {
            if let Some(page) = page {
                page.size() == PAGE_SIZE_2M
                    && pt
                        .remap_huge(range.start, page.paddr(), flags, PageSize::Size2M)
                        .is_ok()
            } else {
                false
            }
        });
        if reused_huge {
            return Ok(());
        }

        for fault_addr in fault_addrs {
            let populate_result = self.populate(
                VirtAddrRange::from_start_size(fault_addr, PAGE_SIZE_4K),
                flags,
                flags - MappingFlags::WRITE,
                pt,
            );
            match populate_result {
                Ok((_, callback)) => {
                    if let Some(cb) = callback {
                        callbacks.as_mut().unwrap().push(cb);
                    }
                }
                Err(err) => {
                    warn!("Failed to populate page for {fault_addr:?} ({flags:?}): {err}");
                }
            };
        }

        let new_pa = self.0.cache.replace_with_huge_page(start_pn)?;
        pt.remap_huge(range.start, new_pa, flags, PageSize::Size2M)?;
        Ok(())
    }

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

    fn transparent_hugepage_enabled(&self) -> bool {
        let inner = &self.0;
        inner.transparent_hugepage_enabled()
    }

    fn set_vma_flag(&self, flag: VmaFlags) {
        let inner = &self.0;
        inner.set_vma_flag(flag);
    }

    fn clear_vma_flag(&self, flag: VmaFlags) {
        let inner = &self.0;
        inner.clear_vma_flag(flag);
    }

    fn contain_vma_flag(&self, flag: VmaFlags) -> bool {
        let inner = &self.0;
        inner.vma_flags.lock().contains(flag)
    }


    fn page_size(&self) -> PageSize {
        PageSize::Size4K
    }

    fn map(&self, _range: VirtAddrRange, flags: MappingFlags, _pt: &mut PageTableMut) -> AxResult {
        self.check_flags(flags)
    }

    fn unmap(&self, range: VirtAddrRange, pt: &mut PageTableMut) -> AxResult {
            if !range.start.is_aligned(PAGE_SIZE_4K)
            || !range.end.is_aligned(PAGE_SIZE_4K)
        {
            return Err(AxError::InvalidInput);
        }

        let mut va = range.start;
        let end = range.end;
        while va < end {
            match pt.query(va) {
                Ok((paddr, _, page_size)) => {
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

    fn on_protect(
        &self,
        _range: VirtAddrRange,
        new_flags: MappingFlags,
        _pt: &mut PageTableMut,
    ) -> AxResult {
        self.check_flags(new_flags)
    }

    
    fn populate(
        &self,
        range: VirtAddrRange,
        flags: MappingFlags,
        access_flags: MappingFlags,
        pt: &mut PageTableMut,
    ) -> AxResult<(usize, Option<Box<dyn FnOnce(&mut AddrSpace)>>)> {
        if !range.start.is_aligned_4k()
            || !range.end.is_aligned_4k() 
        {
            return Err(AxError::InvalidInput);
        }

        let offset = range.start - self.0.start;
        // access that lies beyond the
        // end of the mapped file is not allowed
        let file_len = self.0.cache.len()?;
        if offset >= file_len as usize {
            return Err(AxError::BadAddress);
        }
        let mut pages = 0;
        let mut to_be_evicted = Vec::new();
        let start_page = (offset / PAGE_SIZE_4K) as u32 + self.0.offset_page;
        let mut start_chunk = 0;
        let mut pn = start_page;
        let mut va = range.start;
        let end = range.end;
        while va < end {        
                match pt.query(va) {
                    Ok((paddr, page_flags, page_size)) => {
                        if access_flags.contains(MappingFlags::WRITE)
                            && !page_flags.contains(MappingFlags::WRITE)
                        {
                            let in_memory = self.0.cache.in_memory();
                            // For non-memory files we only support read-only mappings, so this path
                            // handles them as 4KiB pages (no writable THP for file-backed mappings).
                            self.0.cache.with_page(pn, |page| {
                                if !in_memory {
                                    page.expect("page should be present").mark_dirty();
                                }
                                pt.remap(va, paddr, flags)?;
                                pages += 1;
                                pn += 1;
                                va += PAGE_SIZE_4K;
                                AxResult::Ok(())
                            })?;
                        } else if page_flags.contains(access_flags) {
                            let base_va = va.align_down(page_size);
                            let off = va - base_va;
                            va += page_size as usize - off;
                            pn += (page_size as usize - off) as u32 / PAGE_SIZE_4K as u32;
                        }
                    }
                    // If the page is not mapped, try map it.
                    Err(PagingError::NotMapped) => {
                        let map_flags = if self.0.cache.in_memory() {
                            // For in memory files, we don't need to (and also
                            // musn't) mark them dirty, so we can use the original
                            // flags.
                            flags
                        } else {
                            flags - MappingFlags::WRITE
                        };
                        let (opt_num, size) = self.0.cache.locate_chunk(start_chunk, pn);
                        let mut base_va = va;
                        let mut page_size = PAGE_SIZE_4K;
                        let mut base_pn = pn;
                        if let Some(chunk_num) = opt_num {
                            base_pn = chunk_num;
                            start_chunk = chunk_num;
                            page_size = size;
                        }
                        self.0.cache.with_page_or_insert(base_pn, page_size, |page, evicted| {
                            if let Some((pn, _)) = evicted {
                                to_be_evicted.push(pn);
                            }
                            if va.is_aligned(page_size) 
                            && range.contains(va) 
                            && (range.end - va) >= page_size 
                            {
                                let size = if page_size == PAGE_SIZE_4K {
                                    PageSize::Size4K
                                } else {
                                    PageSize::Size2M
                                };
                                pt.map(va, page.paddr(), size, map_flags)?;
                                pages += 1;
                                va += page_size;
                                pn += page_size as u32 / PAGE_SIZE_4K as u32;
                                start_chunk = pn;
                                return Ok(())
                            }

                            let pa = page.paddr() + (pn - base_pn) as usize * PAGE_SIZE_4K;
                            pt.map(va, pa, PageSize::Size4K, map_flags)?;
                            pages += 1;
                            va += PAGE_SIZE_4K;
                            pn += 1;
                            // The `start_chunk` remains unchanged.
                            Ok(())
                        })?;
                    }
                    Err(_) => return Err(AxError::BadAddress),
            }
        }
        Ok((
            pages,
            if to_be_evicted.is_empty() {
                None
            } else {
                let inner = self.0.clone();
                Some(Box::new(move |aspace: &mut AddrSpace| {
                    for pn in to_be_evicted {
                        inner.on_evict(pn, aspace);
                    }
                }))
            },
        ))
    }

    fn clone_map(
        &self,
        _range: VirtAddrRange,
        _flags: MappingFlags,
        _old_pt: &mut PageTableMut,
        _new_pt: &mut PageTableMut,
        new_aspace: &Arc<Mutex<AddrSpace>>,
    ) -> AxResult<Backend> {
        let inner = Arc::new(FileBackendInner {
            start: self.0.start,
            cache: self.0.cache.clone(),
            flags: self.0.flags,
            offset_page: self.0.offset_page,
            handle: AtomicUsize::new(0),
            futex_handle: self.0.futex_handle.clone(),
            vma_flags: VmaFlags::empty().into(),
        });
        inner.register_listener(new_aspace);
        Ok(Backend::File(FileBackend(inner)))
    }
}

impl Backend {
    pub fn new_file(
        start: VirtAddr,
        cache: CachedFile,
        flags: FileFlags,
        offset: usize,
        aspace: &Arc<Mutex<AddrSpace>>,
    ) -> Self {
        // TODO !!!! not offset / PAGE_SIZE_4K
        let offset_page = (offset / PAGE_SIZE_4K) as u32;
        let inner = Arc::new(FileBackendInner {
            start,
            cache,
            flags,
            offset_page,
            handle: AtomicUsize::new(0),
            futex_handle: Arc::new(()),
            vma_flags: VmaFlags::empty().into(),
        });
        inner.register_listener(aspace);
        Self::File(FileBackend(inner))
    }
}

//! Memory mapping backends.
use alloc::{boxed::Box, sync::Arc, vec::Vec};

use axalloc::{UsageKind, global_allocator};
use axerrno::{AxError, AxResult};
use axfs_ng::CachedFile;
use axhal::{
    mem::{phys_to_virt, virt_to_phys},
    paging::{MappingFlags, PageSize, PageTable, PageTableMut},
};
use axsync::Mutex;
use enum_dispatch::enum_dispatch;
use memory_addr::{DynPageIter, PAGE_SIZE_4K, PhysAddr, VirtAddr, VirtAddrRange};
use memory_set::MappingBackend;
use bitflags::bitflags;

pub mod cow;
pub mod file;
pub mod linear;
pub mod shared;

use page_table_multiarch::PagingError;
pub use shared::{current_shmem_thp_policy, set_shmem_thp_policy};
pub use cow::{current_thp_policy, frame_ref_count, set_thp_policy};

use crate::AddrSpace;

fn divide_page(size: usize, page_size: PageSize) -> usize {
    assert!(page_size.is_aligned(size), "unaligned");
    size >> (page_size as usize).trailing_zeros()
}

fn alloc_frame(zeroed: bool, size: PageSize) -> AxResult<PhysAddr> {
    let page_size = size as usize;
    let num_pages = page_size / PAGE_SIZE_4K;
    let vaddr =
        VirtAddr::from(global_allocator().alloc_pages(num_pages, page_size, UsageKind::UserMem)?);
    if zeroed {
        unsafe { core::ptr::write_bytes(vaddr.as_mut_ptr(), 0, page_size) };
    }
    let paddr = virt_to_phys(vaddr);

    Ok(paddr)
}

fn dealloc_frame(frame: PhysAddr, align: PageSize) {
    let vaddr = phys_to_virt(frame);
    let page_size: usize = align.into();
    let num_pages = page_size / PAGE_SIZE_4K;
    global_allocator().dealloc_pages(vaddr.as_usize(), num_pages, UsageKind::UserMem);
}

fn pages_in(range: VirtAddrRange, align: PageSize) -> AxResult<DynPageIter<VirtAddr>> {
    DynPageIter::new(range.start, range.end, align as usize).ok_or(AxError::InvalidInput)
}

fn register_cache_listener<T>(
    cache: &CachedFile,
    backend: &Arc<T>,
    aspace: &Arc<Mutex<AddrSpace>>,
    on_evict: impl Fn(&Arc<T>, u32, &mut AddrSpace) + Send + Sync + 'static,
) -> usize 
where 
    T: Send + Sync + 'static,
{
    let backend_w = Arc::downgrade(backend);
    let aspace_w = Arc::downgrade(aspace);
    cache.add_evict_listener(move |pn, _page| {
        let Some(backend) = backend_w.upgrade() else { return; };
        let Some(aspace) = aspace_w.upgrade() else { return; };
        let Some(mut aspace) = aspace.try_lock() else { return; };
        on_evict(&backend, pn, &mut aspace);
    })
}

trait EvictMatch {
    fn matches_backend(&self, backend: &Backend) -> bool;
}

impl EvictMatch for file::FileBackendInner {
    fn matches_backend(&self, backend: &Backend) -> bool {
        matches!(backend, Backend::File(file) if file.ptr_eq_inner(self))
    }
}

impl EvictMatch for shared::SharedBackendInner {
    fn matches_backend(&self, backend: &Backend) -> bool {
        matches!(backend, Backend::Shared(shared) if shared.ptr_eq_inner(self))
    }
}

fn on_evict<T>(
    backend: &Arc<T>, 
    vaddr: VirtAddr,
    aspace: &mut AddrSpace
)
where
    T: EvictMatch,
{
    if !aspace
        .find_area(vaddr)
        .is_some_and(|it| backend.matches_backend(it.backend()))
    {
        // Ignore if the page is not controlled by this file mapping.
        return;
    }

    let pt = aspace.page_table_mut();
    match pt.modify().unmap(vaddr) {
        Ok(_) | Err(PagingError::NotMapped) => {}
        Err(err) => {
            warn!("Failed to unmap page {:?}: {:?}", vaddr, err);
        }
    }
}

#[enum_dispatch]
pub trait BackendOps {
    /// Returns the page size of the backend.
    fn page_size(&self) -> PageSize;

    /// Map a memory region.
    fn map(&self, range: VirtAddrRange, flags: MappingFlags, pt: &mut PageTableMut) -> AxResult;

    /// Unmap a memory region.
    fn unmap(&self, range: VirtAddrRange, pt: &mut PageTableMut) -> AxResult;

    /// Called before a memory region is protected.
    fn on_protect(
        &self,
        _range: VirtAddrRange,
        _new_flags: MappingFlags,
        _pt: &mut PageTableMut,
    ) -> AxResult {
        Ok(())
    }

    /// Populate a memory region and return how many pages now satisfy
    /// `access_flags`.
    ///
    /// If another thread has already mapped the page with sufficient permissions,
    /// treat it as populated.
    fn populate(
        &self,
        _range: VirtAddrRange,
        _flags: MappingFlags,
        _access_flags: MappingFlags,
        _pt: &mut PageTableMut,
    ) -> AxResult<(usize, Option<Box<dyn FnOnce(&mut AddrSpace)>>)> {
        Ok((0, None))
    }
    
    /// Duplicates this mapping for use in a different page table.
    ///
    /// This differs from `clone`, which is designed for splitting a mapping
    /// within the same table.
    ///
    /// [`BackendOps::map`] will be latter called to the returned backend.
    fn clone_map(
        &self,
        range: VirtAddrRange,
        flags: MappingFlags,
        old_pt: &mut PageTableMut,
        new_pt: &mut PageTableMut,
        new_aspace: &Arc<Mutex<AddrSpace>>,
    ) -> AxResult<Backend>;

    /// Set per-VMA THP-related flags (e.g. VM_HUGEPAGE / VM_NOHUGEPAGE).
    fn set_vma_flag(&self, flag: VmaFlags);

    /// Clear per-VMA THP-related flags.
    fn clear_vma_flag(&self, flag: VmaFlags);

    /// Contain the per-VMA THP-related flag 
    fn contain_vma_flag(&self, flag: VmaFlags) -> bool;

    fn transparent_hugepage_enabled(&self) -> bool;

    /// Backend-specific THP collapse attempt on a single 2 MiB window.
    ///
    /// - `range` is a 2 MiB‑aligned virtual address window.
    /// - `pt` is the page table to operate on.
    /// - `max_ptes_none` / `max_ptes_shared` are collapse heuristics.
    /// - `pages_scanned` is increased by the number of 4 KiB pages examined,
    ///   and must never exceed `pages_to_scan`.
    ///
    /// Returns `Ok(true)` if a collapse actually happened, `Ok(false)` if the
    /// backend decided not to collapse this window, and `Err(_)` on failure.
    fn try_collapse_page(
        &self,
        range: VirtAddrRange,
        pt: &mut PageTableMut,
        max_ptes_none: usize,
        max_ptes_shared: usize,
        pages_scanned: &mut usize,
        pages_to_scan: usize,
    ) -> AxResult<bool>;

    fn collapse_page(
        &self,
        range: VirtAddrRange,
        pt: &mut PageTableMut,
        fault_in: bool,
        callbacks: Option<&mut Vec<Box<dyn FnOnce(&mut AddrSpace)>>>,
    ) -> AxResult;
}

/// A unified enum type for different memory mapping backends.
#[derive(Clone)]
#[enum_dispatch(BackendOps)]
pub enum Backend {
    Linear(linear::LinearBackend),
    Cow(cow::CowBackend),
    Shared(shared::SharedBackend),
    File(file::FileBackend),
}

impl MappingBackend for Backend {
    type Addr = VirtAddr;
    type Flags = MappingFlags;
    type PageTable = PageTable;

    fn map(&self, start: VirtAddr, size: usize, flags: MappingFlags, pt: &mut PageTable) -> bool {
        let range = VirtAddrRange::from_start_size(start, size);
        if let Err(err) = BackendOps::map(self, range, flags, &mut pt.modify()) {
            warn!("Failed to map area: {:?}", err);
            false
        } else {
            true
        }
    }

    fn unmap(&self, start: VirtAddr, size: usize, pt: &mut PageTable) -> bool {
        let range = VirtAddrRange::from_start_size(start, size);
        if let Err(err) = BackendOps::unmap(self, range, &mut pt.modify()) {
            warn!("Failed to unmap area: {:?}", err);
            false
        } else {
            true
        }
    }

    fn protect(
        &self,
        start: Self::Addr,
        size: usize,
        new_flags: Self::Flags,
        pt: &mut Self::PageTable,
    ) -> bool {
        pt.modify().protect_region(start, size, new_flags).is_ok()
    }
}


bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct VmaFlags: usize {
        /// Currently indicate it is a stack memory
        const VM_STACK	 = 0x0000_0100;
        /// Page-ranges managed without `struct page`, pure PFNs (Linux VM_PFNMAP).
        const VM_PFNMAP      = 0x0000_0400;
        /// Memory-mapped I/O or similar (Linux VM_IO).
        const VM_IO          = 0x0000_4000;
        /// Cannot expand with mremap() (Linux VM_DONTEXPAND).
        const VM_DONTEXPAND  = 0x0004_0000;
        /// HugeTLB mapping (Linux VM_HUGETLB).
        const VM_HUGETLB     = 0x0040_0000;
        /// Mapping may contain both struct page and pure PFN pages (Linux VM_MIXEDMAP).
        const VM_MIXEDMAP    = 0x1000_0000;
        /// madvise(MADV_HUGEPAGE): mark VMA as THP candidate (Linux VM_HUGEPAGE).
        const VM_HUGEPAGE    = 0x2000_0000;
        /// madvise(MADV_NOHUGEPAGE): forbid THP on this VMA (Linux VM_NOHUGEPAGE).
        const VM_NOHUGEPAGE  = 0x4000_0000;
    }
}

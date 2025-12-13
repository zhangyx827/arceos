use alloc::{boxed::Box, sync::Arc, vec::Vec};

use axerrno::AxResult;
use axhal::paging::{MappingFlags, PageSize, PageTableMut};
use axsync::Mutex;
use memory_addr::{PhysAddr, PhysAddrRange, VirtAddr, VirtAddrRange};

use crate::{
    AddrSpace,
    backend::{Backend, BackendOps, VmaFlags},
};

/// Linear mapping backend.
///
/// The offset between the virtual address and the physical address is
/// constant, which is specified by `pa_va_offset`. For example, the virtual
/// address `vaddr` is mapped to the physical address `vaddr - pa_va_offset`.
#[derive(Clone)]
pub struct LinearBackend {
    offset: isize,
}

impl LinearBackend {
    fn pa(&self, va: VirtAddr) -> PhysAddr {
        PhysAddr::from((va.as_usize() as isize - self.offset) as usize)
    }
}

impl BackendOps for LinearBackend {
    fn try_collapse_page(
        &self,
        _range: VirtAddrRange,
        _pt: &mut PageTableMut,
        _max_ptes_none: usize,
        _max_ptes_shared: usize,
        _pages_scanned: &mut usize,
        _pages_to_scan: usize,
    ) -> AxResult<bool> {
        // Linear backend does not support THP collapse.
        Ok(false)
    }

    fn collapse_page(
        &self,
        _range: VirtAddrRange,
        _pt: &mut PageTableMut,
        _fault_in: bool,
        _callbacks: Option<&mut Vec<Box<dyn FnOnce(&mut AddrSpace)>>>,
    ) -> AxResult {
        // Linear backend does not support THP collapse.
        Ok(())
    }

    fn set_vma_flag(&self, _vma_flags: VmaFlags) {
        // Linear mappings do not participate in THP policy.
    }

    fn clear_vma_flag(&self, _vma_flags: VmaFlags) {
        // Linear mappings do not participate in THP policy.
    }

    fn transparent_hugepage_enabled(&self) -> bool {
        false
    }

    fn contain_vma_flag(&self, _flag: VmaFlags) -> bool {
        // Linear mappings do not participate in THP policy.
        false
    }


    fn page_size(&self) -> PageSize {
        PageSize::Size4K
    }

    fn map(&self, range: VirtAddrRange, flags: MappingFlags, pt: &mut PageTableMut) -> AxResult {
        let pa_range = PhysAddrRange::from_start_size(self.pa(range.start), range.size());
        debug!("Linear::map: {range:?} -> {pa_range:?} {flags:?}");
        pt.map_region(range.start, |va| self.pa(va), range.size(), flags, false)?;
        Ok(())
    }

    fn unmap(&self, range: VirtAddrRange, pt: &mut PageTableMut) -> AxResult {
        let pa_range = PhysAddrRange::from_start_size(self.pa(range.start), range.size());
        debug!("Linear::unmap: {range:?} -> {pa_range:?}");
        pt.unmap_region(range.start, range.size())?;
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
        Ok(Backend::Linear(self.clone()))
    }
}

impl Backend {
    pub fn new_linear(offset: isize) -> Self {
        Self::Linear(LinearBackend { offset })
    }
}

// Derived from tachometer-writer via NVIDIA/srt-slurm (Apache-2.0); modified for niu-lai-viz. See NOTICE.
pub mod compaction;
pub mod error;
pub mod writer;

pub use compaction::{compact_and_upload, periodic_compact_and_sync};
pub use error::{NoMoreError, Result};
pub use writer::{DatasetWriter, Row};

use std::alloc::LayoutError;
use std::fmt;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    OutOfMemory,
    PointerUnderflow,
    Layout(std::alloc::LayoutError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OutOfMemory => write!(f, "Out of Memory"),
            Error::Layout(e) => write!(f, "Layout Error: {}", e),
            Error::PointerUnderflow => write!(f, "Pointer underflow"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Layout(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LayoutError> for Error {
    fn from(error: LayoutError) -> Self {
        Error::Layout(error)
    }
}

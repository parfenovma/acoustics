import dataclasses
import typing



class ConfigProtocol(typing.Protocol):
    """Defines the base contract for configuration objects.

    This protocol serves as a foundation for more specific configuration types.
    Different problems require distinct configurations; for instance, a
    frequency-based solution does not require a pulse definition, while a
    time-based solution does.

    It is intended for concrete configuration protocols to be derived from
    this base and defined within their respective solution modules. The primary
    benefit of this approach is that static analysis tools can provide
    immediate feedback if a configuration object does not adhere to the
    contract required by a solution.
    """
    c0: float  # Speed of sound [m/s]


    # geometric parameters
    width: float
    height: float
    A: float
    num_waves: int
    num_points_boundary: int
    n_elem: int
    deg: int = 2
    boundary_type: typing.Literal['rigid', 'wavy']
    Sx: float
    Sy: float


@dataclasses.dataclass
class TimeDomainSimulationConfig:
    width: float = 1.0; height: float = 1.0; A: float = 0.0; num_waves: int = 0
    num_points_boundary: int = 50; n_elem: int = 64; deg: int = 2
    boundary_type: str = 'rigid'

    # Time-domain specific parameters
    t_end: float = 0.006  # End time of simulation [s]
    dt: float = 2e-6      # Time step [s] - MUST satisfy CFL condition

    # # Source parameters (for Ricker pulse)
    # source_freq: float = 2000.0 # Center frequency of the Ricker pulse [Hz]
    # source_delay: float = 0.0005 # Time delay for the pulse peak [s]

    # Hamming pulse 
    pulse_duration: float = 0.001 # Hamming window duration [s]
    pulse_carrier_freq: float = 2000.0 # main sinusoidal frequency [Hz]
    Sx: float = 0.5; Sy: float = 0.5
    
    # Physical constants
    c0: float = 340.0

    # PML parameters
    pml_thickness: float = 0.1 # Толщина PML слоя
    pml_sigma_max: float = 5000.0 # Максимальное значение затухания

@dataclasses.dataclass
class FrequencyDomainSimulationConfig:
    width: float = 1.0
    height: float = 1.0
    A: float = 0.1
    num_waves: int = 3
    num_points_boundary: int = 50
    n_elem: int = 64
    deg: int = 2
    freq: float = 800.0
    Sx: float = 0.1
    Sy: float = 0.5
    boundary_type: str = 'wavy'  # 'wavy' or 'rigid'

    # Physical constants
    c0: float = 340.0
    rho_0: float = 1.225

    # Source properties
    Q: float = 0.0001
    alfa: float = 0.015 # for Gaussian source approximation

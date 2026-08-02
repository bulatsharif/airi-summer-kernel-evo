"""Domain errors raised by the interactive evolution controller."""


class KernelEvoAgentError(RuntimeError):
    """Base error suitable for a concise CLI diagnostic."""


class RunNotFoundError(KernelEvoAgentError):
    """The requested run id does not exist under the selected runs directory."""


class InvalidTransitionError(KernelEvoAgentError):
    """A command violated the barrier-synchronized run lifecycle."""


class ConfigurationError(KernelEvoAgentError):
    """An agent-run configuration is missing or inconsistent."""

"""
Base class for all LLM processing steps.

Steps are reusable, composable transformation units that operate on GlobalState.
They use the input_map pattern to decouple from other steps.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel

from llm_etl.core.state import GlobalState


class AbstractBaseStep(ABC):
    """
    Abstract base class for all pipeline steps.

    Steps operate on GlobalState using an input_map pattern:
    1. The input_map defines lambdas that extract data from GlobalState
    2. The step executes using the mapped inputs (doesn't know about GlobalState structure)
    3. Results are stored back in GlobalState.processed[output_key]

    This decoupling allows steps to be reused and composed without dependencies.

    Example:
        >>> class MyStep(AbstractBaseStep):
        ...     def execute(self, mapped_input: dict, llm_client) -> BaseModel:
        ...         text = mapped_input["text"]
        ...         # Call LLM, return Pydantic model
        ...         return MyOutputSchema(result="processed")
        ...
        >>> step = MyStep(
        ...     name="my_step",
        ...     input_map={"text": lambda s: s.raw["note"]},
        ...     output_key="my_output"
        ... )
        >>> state = GlobalState(pk="123", raw={"note": "Patient presents..."})
        >>> state = step.run(state, llm_client)
        >>> state.processed["my_output"]["result"]
        'processed'
        >>> "my_step" in state.log
        True
    """

    def __init__(
        self,
        name: str,
        input_map: dict[str, Callable[[GlobalState], Any]],
        output_key: str,
    ):
        """
        Initialize a step with its configuration.

        Args:
            name: Unique identifier for this step (used in logs and errors)
            input_map: Dictionary mapping parameter names to lambdas that extract
                      values from GlobalState. Keys become parameters in execute().
            output_key: Key under which to store this step's output in
                       GlobalState.processed

        Example:
            >>> step = MyStep(
            ...     name="classifier",
            ...     input_map={
            ...         "text": lambda s: s.raw["clinical_note"],
            ...         "context": lambda s: f"Dept: {s.raw['department']}"
            ...     },
            ...     output_key="visit_type"
            ... )
        """
        self.name = name
        self.input_map = input_map
        self.output_key = output_key

    def _apply_input_map(self, state: GlobalState) -> dict[str, Any]:
        """
        Apply the input_map to extract values from GlobalState.

        Executes each lambda in input_map with the state, returning a dict
        with the same keys but resolved values.

        Args:
            state: The GlobalState to extract values from

        Returns:
            Dictionary with same keys as input_map, values resolved from state

        Example:
            >>> state = GlobalState(pk="123", raw={"note": "text", "age": 45})
            >>> step = MyStep(
            ...     name="test",
            ...     input_map={
            ...         "text": lambda s: s.raw["note"],
            ...         "context": lambda s: f"Age: {s.raw['age']}"
            ...     },
            ...     output_key="output"
            ... )
            >>> mapped = step._apply_input_map(state)
            >>> mapped["text"]
            'text'
            >>> mapped["context"]
            'Age: 45'
        """
        return {key: func(state) for key, func in self.input_map.items()}

    @abstractmethod
    def execute(self, mapped_input: dict[str, Any], llm_client: Any, pk: str) -> BaseModel:
        """
        Execute the step's core logic using mapped inputs.

        Subclasses must implement this method to define their transformation logic.
        The method receives already-mapped inputs (not the raw GlobalState) and
        must return a Pydantic BaseModel that will be stored in GlobalState.processed.

        Args:
            mapped_input: Dictionary of inputs resolved from input_map
            llm_client: LLM client instance for making API calls
            pk: Primary key of the record being processed (for logging)

        Returns:
            Pydantic BaseModel containing the step's output

        Raises:
            StepExecutionError: If step logic fails
            LLMValidationError: If LLM output fails validation

        Example:
            >>> def execute(self, mapped_input: dict, llm_client, pk: str) -> MyOutputSchema:
            ...     text = mapped_input["text"]
            ...     response = llm_client.call(prompt=f"Analyze: {text}")
            ...     return MyOutputSchema.model_validate(response)
        """
        pass

    def run(self, state: GlobalState, llm_client: Any) -> GlobalState:
        """
        Execute the step on a GlobalState, updating it with results.

        This is the main entry point for running a step. It:
        1. Applies the input_map to extract values from state
        2. Calls execute() with the mapped inputs
        3. Stores the result in state.processed[output_key]
        4. Appends the step name to state.log
        5. Returns the updated state

        Args:
            state: The GlobalState to process
            llm_client: LLM client instance for making API calls

        Returns:
            The updated GlobalState with results in .processed and step logged

        Example:
            >>> state = GlobalState(pk="enc_123", raw={"note": "Patient presents..."})
            >>> step = MyStep(
            ...     name="classifier",
            ...     input_map={"text": lambda s: s.raw["note"]},
            ...     output_key="category"
            ... )
            >>> state = step.run(state, llm_client)
            >>> "category" in state.processed
            True
            >>> "classifier" in state.log
            True
        """
        # 1. Apply input_map to extract values from state
        mapped_input = self._apply_input_map(state)

        # 2. Execute the step's core logic
        result = self.execute(mapped_input, llm_client, pk=state.pk)

        # 3. Store result as dict in processed
        state.processed[self.output_key] = result.model_dump()

        # 4. Log that this step ran
        state.log.append(self.name)

        # 5. Return updated state
        return state

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"output_key={self.output_key!r})"
        )

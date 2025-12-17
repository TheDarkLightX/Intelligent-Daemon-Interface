"""Hypothesis strategies for Tau spec generation and testing.

Provides deterministic, bounded generators for:
- StreamConfig (input/output streams with types)
- LogicBlock (pattern blocks with valid input references)
- AgentSchema (complete agent specifications)

Design Constraints:
- All generators are seeded for reproducibility
- Bounded complexity (max streams, max blocks, max steps)
- Cross-reference validation (inputs reference existing streams)
- Type-safe generation (sbf vs bv constraints)

Usage:
    from idi.devkit.tau_factory.tests.strategies import agent_schema_strategy
    
    @given(schema=agent_schema_strategy())
    def test_schema_generates_valid_spec(schema):
        ...
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from hypothesis import assume, settings
    from hypothesis import strategies as st
    from hypothesis.strategies import SearchStrategy

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    st = None  # type: ignore
    SearchStrategy = Any  # type: ignore

    def assume(condition: bool) -> None:  # type: ignore
        pass

from idi.devkit.tau_factory.schema import (
    AgentSchema,
    LogicBlock,
    StreamConfig,
)


# =============================================================================
# Constants / Bounds
# =============================================================================

MAX_STREAMS = 8
MAX_LOGIC_BLOCKS = 6
MAX_NUM_STEPS = 20
MIN_NUM_STEPS = 1
MAX_BV_WIDTH = 16
MIN_BV_WIDTH = 1
MAX_NAME_LEN = 32

# Safe pattern subset (verified working, single-input compatible)
SAFE_PATTERNS = (
    "passthrough",
    "vote",
    "majority",
    "unanimous",
    "counter",
)

# Patterns requiring 2+ inputs
MULTI_INPUT_PATTERNS = (
    "fsm",  # Requires buy/sell inputs
)

# Extended patterns (may have edge cases)
EXTENDED_PATTERNS = (
    "accumulator",
    "quorum",
    "custom",
    "weighted_vote",
    "time_lock",
    "mode_switch",
)


# =============================================================================
# Primitive Strategies
# =============================================================================

if HAS_HYPOTHESIS:
    # Stream/variable names: alphanumeric, start with letter
    stream_name_strategy: SearchStrategy[str] = st.from_regex(
        r"[a-z][a-z0-9_]{2,15}", fullmatch=True
    )

    # Agent names
    agent_name_strategy: SearchStrategy[str] = st.from_regex(
        r"[A-Z][A-Za-z0-9_]{2,24}", fullmatch=True
    )

    # Stream types
    stream_type_strategy: SearchStrategy[str] = st.sampled_from(["sbf", "bv"])

    # BV width
    bv_width_strategy: SearchStrategy[int] = st.integers(
        min_value=MIN_BV_WIDTH, max_value=MAX_BV_WIDTH
    )

    # Strategy type
    strategy_type_strategy: SearchStrategy[str] = st.sampled_from([
        "momentum", "mean_reversion", "regime_aware", "custom"
    ])

    # Number of steps
    num_steps_strategy: SearchStrategy[int] = st.integers(
        min_value=MIN_NUM_STEPS, max_value=MAX_NUM_STEPS
    )

    # Safe patterns only (for reliable generation)
    safe_pattern_strategy: SearchStrategy[str] = st.sampled_from(SAFE_PATTERNS)

    # Extended patterns (includes more complex ones)
    extended_pattern_strategy: SearchStrategy[str] = st.sampled_from(
        SAFE_PATTERNS + EXTENDED_PATTERNS
    )


# =============================================================================
# StreamConfig Strategies
# =============================================================================

if HAS_HYPOTHESIS:

    @st.composite
    def stream_config_strategy(
        draw: st.DrawFn,
        is_input: bool = True,
        name: str | None = None,
    ) -> StreamConfig:
        """Generate a valid StreamConfig."""
        stream_name = name or draw(stream_name_strategy)
        stream_type = draw(stream_type_strategy)

        if stream_type == "bv":
            width = draw(bv_width_strategy)
        else:
            width = 8  # Ignored for sbf

        return StreamConfig(
            name=stream_name,
            stream_type=stream_type,
            width=width,
            is_input=is_input,
        )

    @st.composite
    def input_stream_strategy(draw: st.DrawFn) -> StreamConfig:
        """Generate an input stream."""
        return draw(stream_config_strategy(is_input=True))

    @st.composite
    def output_stream_strategy(draw: st.DrawFn) -> StreamConfig:
        """Generate an output stream."""
        return draw(stream_config_strategy(is_input=False))


# =============================================================================
# LogicBlock Strategies
# =============================================================================

if HAS_HYPOTHESIS:

    @st.composite
    def passthrough_block_strategy(
        draw: st.DrawFn,
        input_names: List[str],
        output_name: str,
    ) -> LogicBlock:
        """Generate a passthrough logic block."""
        assume(len(input_names) >= 1)
        selected_input = draw(st.sampled_from(input_names))
        return LogicBlock(
            pattern="passthrough",
            inputs=(selected_input,),
            output=output_name,
            params={},
        )

    @st.composite
    def vote_block_strategy(
        draw: st.DrawFn,
        input_names: List[str],
        output_name: str,
    ) -> LogicBlock:
        """Generate a vote/majority logic block."""
        assume(len(input_names) >= 2)
        # Select 2-3 inputs for voting
        num_inputs = draw(st.integers(min_value=2, max_value=min(3, len(input_names))))
        selected = draw(st.permutations(input_names).map(lambda x: x[:num_inputs]))
        pattern = draw(st.sampled_from(["vote", "majority", "unanimous"]))
        return LogicBlock(
            pattern=pattern,
            inputs=tuple(selected),
            output=output_name,
            params={},
        )

    @st.composite
    def fsm_block_strategy(
        draw: st.DrawFn,
        input_names: List[str],
        output_name: str,
    ) -> LogicBlock:
        """Generate an FSM logic block (requires 2+ inputs for buy/sell)."""
        assume(len(input_names) >= 2)
        # FSM requires at least 2 inputs (buy_signal, sell_signal)
        num_inputs = draw(st.integers(min_value=2, max_value=min(3, len(input_names))))
        selected = draw(st.permutations(input_names).map(lambda x: tuple(x[:num_inputs])))
        num_states = draw(st.integers(min_value=2, max_value=4))
        return LogicBlock(
            pattern="fsm",
            inputs=selected,
            output=output_name,
            params={"num_states": num_states},
        )

    @st.composite
    def counter_block_strategy(
        draw: st.DrawFn,
        input_names: List[str],
        output_name: str,
    ) -> LogicBlock:
        """Generate a counter logic block."""
        assume(len(input_names) >= 1)
        selected_input = draw(st.sampled_from(input_names))
        max_count = draw(st.integers(min_value=2, max_value=255))
        return LogicBlock(
            pattern="counter",
            inputs=(selected_input,),
            output=output_name,
            params={"max_count": max_count},
        )

    @st.composite
    def logic_block_strategy(
        draw: st.DrawFn,
        input_names: List[str],
        output_name: str,
        pattern: str | None = None,
    ) -> LogicBlock:
        """Generate a logic block with valid cross-references."""
        assume(len(input_names) >= 1)

        if pattern is None:
            pattern = draw(safe_pattern_strategy)

        if pattern == "passthrough":
            return draw(passthrough_block_strategy(input_names, output_name))
        elif pattern in ("vote", "majority", "unanimous"):
            if len(input_names) >= 2:
                return draw(vote_block_strategy(input_names, output_name))
            else:
                return draw(passthrough_block_strategy(input_names, output_name))
        elif pattern == "fsm":
            if len(input_names) >= 2:
                return draw(fsm_block_strategy(input_names, output_name))
            else:
                return draw(passthrough_block_strategy(input_names, output_name))
        elif pattern == "counter":
            return draw(counter_block_strategy(input_names, output_name))
        else:
            # Default to passthrough for unknown patterns
            return draw(passthrough_block_strategy(input_names, output_name))


# =============================================================================
# AgentSchema Strategies
# =============================================================================

if HAS_HYPOTHESIS:

    @st.composite
    def minimal_agent_schema_strategy(draw: st.DrawFn) -> AgentSchema:
        """Generate minimal valid AgentSchema (1 input, 1 output, 1 block)."""
        name = draw(agent_name_strategy)
        strategy = draw(strategy_type_strategy)

        # Single input
        input_name = draw(stream_name_strategy)
        input_stream = StreamConfig(
            name=input_name,
            stream_type="sbf",
            width=8,
            is_input=True,
        )

        # Single output
        output_name = draw(stream_name_strategy.filter(lambda n: n != input_name))
        output_stream = StreamConfig(
            name=output_name,
            stream_type="sbf",
            width=8,
            is_input=False,
        )

        # Single passthrough block
        block = LogicBlock(
            pattern="passthrough",
            inputs=(input_name,),
            output=output_name,
            params={},
        )

        return AgentSchema(
            name=name,
            strategy=strategy,
            streams=(input_stream, output_stream),
            logic_blocks=(block,),
            num_steps=draw(num_steps_strategy),
            include_mirrors=False,
            descriptive_names=False,
        )

    @st.composite
    def agent_schema_strategy(
        draw: st.DrawFn,
        max_inputs: int = 4,
        max_outputs: int = 3,
        safe_only: bool = True,
    ) -> AgentSchema:
        """Generate a valid AgentSchema with cross-reference consistency.
        
        Invariants:
        - All logic block inputs reference declared input streams
        - All logic block outputs reference declared output streams
        - No duplicate stream names
        - At least 1 input, 1 output, 1 logic block
        """
        name = draw(agent_name_strategy)
        strategy = draw(strategy_type_strategy)

        # Generate unique input stream names
        num_inputs = draw(st.integers(min_value=1, max_value=max_inputs))
        input_names: List[str] = []
        for _ in range(num_inputs):
            new_name = draw(stream_name_strategy.filter(lambda n: n not in input_names))
            input_names.append(new_name)

        # Generate unique output stream names (different from inputs)
        num_outputs = draw(st.integers(min_value=1, max_value=max_outputs))
        output_names: List[str] = []
        all_names = set(input_names)
        for _ in range(num_outputs):
            new_name = draw(stream_name_strategy.filter(lambda n: n not in all_names))
            output_names.append(new_name)
            all_names.add(new_name)

        # Build input streams (all sbf for simplicity)
        input_streams = [
            StreamConfig(name=n, stream_type="sbf", width=8, is_input=True)
            for n in input_names
        ]

        # Build output streams
        output_streams = [
            StreamConfig(name=n, stream_type="sbf", width=8, is_input=False)
            for n in output_names
        ]

        # Build logic blocks: one per output
        logic_blocks: List[LogicBlock] = []
        for output_name in output_names:
            block = draw(logic_block_strategy(
                input_names=input_names,
                output_name=output_name,
                pattern=draw(safe_pattern_strategy) if safe_only else None,
            ))
            logic_blocks.append(block)

        return AgentSchema(
            name=name,
            strategy=strategy,
            streams=tuple(input_streams + output_streams),
            logic_blocks=tuple(logic_blocks),
            num_steps=draw(num_steps_strategy),
            include_mirrors=draw(st.booleans()),
            descriptive_names=False,  # Use compact names for reliability
        )

    @st.composite
    def bitvector_agent_schema_strategy(draw: st.DrawFn) -> AgentSchema:
        """Generate AgentSchema with bitvector streams."""
        name = draw(agent_name_strategy)
        strategy = draw(strategy_type_strategy)

        # BV input
        input_name = draw(stream_name_strategy)
        input_width = draw(bv_width_strategy)
        input_stream = StreamConfig(
            name=input_name,
            stream_type="bv",
            width=input_width,
            is_input=True,
        )

        # BV output (same width)
        output_name = draw(stream_name_strategy.filter(lambda n: n != input_name))
        output_stream = StreamConfig(
            name=output_name,
            stream_type="bv",
            width=input_width,
            is_input=False,
        )

        # Counter block (works with bv)
        block = LogicBlock(
            pattern="counter",
            inputs=(input_name,),
            output=output_name,
            params={"max_count": (2 ** input_width) - 1},
        )

        return AgentSchema(
            name=name,
            strategy=strategy,
            streams=(input_stream, output_stream),
            logic_blocks=(block,),
            num_steps=draw(num_steps_strategy),
            include_mirrors=False,
            descriptive_names=False,
        )


# =============================================================================
# Adversarial / Edge Case Strategies
# =============================================================================

if HAS_HYPOTHESIS:

    @st.composite
    def empty_name_schema_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate schema dict with empty name (should fail validation)."""
        base = draw(minimal_agent_schema_strategy())
        return {
            "name": "",  # Invalid
            "strategy": base.strategy,
            "streams": base.streams,
            "logic_blocks": base.logic_blocks,
            "num_steps": base.num_steps,
        }

    @st.composite
    def dangling_reference_schema_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate schema with logic block referencing non-existent stream."""
        base = draw(minimal_agent_schema_strategy())
        # Create block referencing non-existent input
        bad_block = LogicBlock(
            pattern="passthrough",
            inputs=("nonexistent_stream",),
            output=base.logic_blocks[0].output,
            params={},
        )
        return {
            "name": base.name,
            "strategy": base.strategy,
            "streams": base.streams,
            "logic_blocks": (bad_block,),
            "num_steps": base.num_steps,
        }


# =============================================================================
# Helpers
# =============================================================================

def make_deterministic_schema(seed: int) -> AgentSchema:
    """Generate deterministic AgentSchema from seed for reproducible tests.
    
    This is a non-Hypothesis helper for creating known-good test cases.
    """
    import hashlib

    # Deterministic names from seed
    h = hashlib.sha256(f"schema_seed_{seed}".encode()).hexdigest()
    name = f"Agent_{h[:8]}"
    input_name = f"in_{h[8:12]}"
    output_name = f"out_{h[12:16]}"

    input_stream = StreamConfig(
        name=input_name,
        stream_type="sbf",
        width=8,
        is_input=True,
    )

    output_stream = StreamConfig(
        name=output_name,
        stream_type="sbf",
        width=8,
        is_input=False,
    )

    block = LogicBlock(
        pattern="passthrough",
        inputs=(input_name,),
        output=output_name,
        params={},
    )

    return AgentSchema(
        name=name,
        strategy="custom",
        streams=(input_stream, output_stream),
        logic_blocks=(block,),
        num_steps=5,
        include_mirrors=False,
        descriptive_names=False,
    )

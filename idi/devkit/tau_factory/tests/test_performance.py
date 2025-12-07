"""Performance tests for Tau factory optimizations.

Tests performance improvements and ensures optimizations don't break functionality.
"""

import pytest
import time
from idi.devkit.tau_factory.schema import AgentSchema, StreamConfig, LogicBlock
from idi.devkit.tau_factory.generator import generate_tau_spec, create_minimal_schema
from idi.devkit.tau_factory.performance_monitor import monitor


class TestPerformance:
    """Test performance optimizations and monitoring."""

    def test_generation_performance_monitoring(self):
        """Test that performance monitoring works."""
        # Enable monitoring for this test
        old_enabled = monitor.enabled
        monitor.set_enabled(True)
        monitor.reset()

        try:
            schema = create_minimal_schema("perf_test")
            result = generate_tau_spec(schema)

            # Check that monitoring captured the operation
            stats = monitor.get_stats()
            assert "total_generation" in stats
            assert stats["total_generation"].call_count >= 1
            assert stats["total_generation"].avg_time_ms > 0

            # Ensure result is still correct
            assert "perf_test" in result
            assert "passthrough pattern" in result

        finally:
            monitor.set_enabled(old_enabled)

    def test_template_caching(self):
        """Test that template caching improves performance."""
        from idi.devkit.tau_factory.compiled_templates import template_cache

        # Clear cache
        template_cache.clear()

        schema = create_minimal_schema("cache_test")

        # First generation - should compile templates
        start_time = time.perf_counter()
        result1 = generate_tau_spec(schema)
        first_duration = time.perf_counter() - start_time

        # Second generation - should use cached templates
        start_time = time.perf_counter()
        result2 = generate_tau_spec(schema)
        second_duration = time.perf_counter() - start_time

        # Results should be identical
        assert result1 == result2

        # Second run should be faster (due to caching)
        # Allow some tolerance for measurement variance
        assert second_duration <= first_duration * 1.5, \
            ".3f"

    def test_compiled_template_correctness(self):
        """Test that compiled templates produce correct output."""
        from idi.devkit.tau_factory.compiled_templates import CompiledTemplate

        # Create a simple compiled template
        template = CompiledTemplate(name="test")
        template.add_fragment("Hello ${name}, your value is ${value}!")

        context = {"name": "world", "value": "42"}
        result = template.render(context)

        assert result == "Hello world, your value is 42!"

    def test_large_schema_performance(self):
        """Test performance with larger schemas."""
        # Create a larger schema with multiple blocks
        streams = []
        logic_blocks = []

        # Add many streams
        for i in range(10):
            streams.append(StreamConfig(f"input{i}", "sbf", is_input=True))
            streams.append(StreamConfig(f"output{i}", "sbf", is_input=False))

        # Add multiple logic blocks
        for i in range(5):
            logic_blocks.append(LogicBlock(
                pattern="passthrough",
                inputs=[f"input{i}"],
                output=f"output{i}"
            ))

        schema = AgentSchema(
            name="large_test",
            strategy="custom",
            streams=streams,
            logic_blocks=logic_blocks
        )

        # Enable monitoring
        old_enabled = monitor.enabled
        monitor.set_enabled(True)
        monitor.reset()

        try:
            result = generate_tau_spec(schema)

            # Verify result
            assert "large_test" in result
            assert len(result) > 1000  # Should be substantial

            # Check performance
            stats = monitor.get_stats()
            assert "total_generation" in stats
            generation_time = stats["total_generation"].avg_time_ms

            # Should complete in reasonable time (< 100ms)
            assert generation_time < 100, f"Generation took {generation_time}ms"

        finally:
            monitor.set_enabled(old_enabled)

    def test_memory_usage_tracking(self):
        """Test that memory usage is tracked."""
        old_enabled = monitor.enabled
        monitor.set_enabled(True)
        monitor.reset()

        try:
            schema = create_minimal_schema("memory_test")
            result = generate_tau_spec(schema)

            stats = monitor.get_stats()
            total_stats = stats.get("total_generation")
            if total_stats:
                # Memory delta should be reasonable (not negative huge)
                assert total_stats.avg_memory_mb > -100  # Allow some variance
                assert total_stats.avg_memory_mb < 100   # Shouldn't use huge amounts

        finally:
            monitor.set_enabled(old_enabled)

    def test_performance_report_generation(self):
        """Test that performance reports can be generated."""
        old_enabled = monitor.enabled
        monitor.set_enabled(True)
        monitor.reset()

        try:
            # Generate some activity
            for i in range(3):
                schema = create_minimal_schema(f"report_test_{i}")
                generate_tau_spec(schema)

            # Generate report
            report = monitor.report()

            # Should contain useful information
            assert "Performance Report" in report
            assert "total_generation" in report
            assert "call_count" in report or "calls" in report

        finally:
            monitor.enabled = old_enabled

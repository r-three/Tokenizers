import re
from unittest.mock import patch

import lm_eval.filters
import pytest
import yaml

from tests.test_experiment import TestExperiment


class TestLmEvalFilters(TestExperiment):
    """Tests for lm-eval-harness filter processing."""

    # Sample YAML configuration to test
    filter_yaml = """
filter_list:
  - name: "answer-with-pound"
    filter:
      - function: "regex"
        regex_pattern: "####\\\\s*(\\\\d+)"
      - function: "take_first"
  - name: "answer-format"
    filter:
      - function: "regex"
        regex_pattern: "answer is:? \\\\$?(\\\\d+)"
      - function: "take_first"
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\\\-?[0-9\\\\.\\\\.\\\\,]+)"
      - function: "take_first"
  - name: "flexible-extract"
    filter:
      - function: "regex"
        group_select: -1
        regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"
      - function: "take_first"
    """

    @pytest.fixture
    def filter_config(self):
        """Parse the filter YAML into a Python object."""
        return yaml.safe_load(self.filter_yaml)

    def test_filter_parsing(self, filter_config):
        """Test that filters parse correctly from YAML."""
        assert "filter_list" in filter_config
        filter_list = filter_config["filter_list"]

        # Check we have the expected number of filters
        assert len(filter_list) == 4

        # Check filter names
        filter_names = [f["name"] for f in filter_list]
        assert "answer-with-pound" in filter_names
        assert "answer-format" in filter_names
        assert "strict-match" in filter_names
        assert "flexible-extract" in filter_names

        # Check specific filter structure
        pound_filter = next(f for f in filter_list if f["name"] == "answer-with-pound")
        assert len(pound_filter["filter"]) == 2
        assert pound_filter["filter"][0]["function"] == "regex"
        assert pound_filter["filter"][0]["regex_pattern"] == "####\\s*(\\d+)"
        assert pound_filter["filter"][1]["function"] == "take_first"

    def test_pound_filter_extraction(self):
        """Test that the pound filter correctly extracts numbers."""
        # Direct regex test
        pattern = re.compile(r"####\s*(\d+)")

        # Test cases
        test_cases = [
            ("After calculating, I get #### 42", "42"),
            ("The answer is #### 123", "123"),
            ("####7", "7"),
            ("#### 42.5", "42"),  # Only extracts the integer part
            ("No match here", None),
        ]

        for input_text, expected in test_cases:
            match = pattern.search(input_text)
            if expected is None:
                assert match is None
            else:
                assert match is not None
                assert match.group(1) == expected

    def test_answer_format_extraction(self):
        """Test that the answer format filter correctly extracts numbers."""
        pattern = re.compile(r"answer is:? \$?(\d+)")

        test_cases = [
            ("The answer is 42", "42"),
            ("The answer is: 123", "123"),
            ("The answer is $99", "99"),
            ("No match here", None),
        ]

        for input_text, expected in test_cases:
            match = pattern.search(input_text)
            if expected is None:
                assert match is None
            else:
                assert match is not None
                assert match.group(1) == expected

    @pytest.mark.parametrize(
        "input_text,expected_result",
        [
            ("After calculating, I get #### 42", "42"),
            ("So the answer is 123", "123"),
            ("The answer is $99", "99"),
            ("This costs $1,234.56", "1,234.56"),
            ("No good match", "No good match"),  # Falls through to identity
        ],
    )
    def test_filter_chain_application(self, filter_config, input_text, expected_result):
        """Test the full chain of filters applied to different inputs."""
        pass
        ## TODO: here actually parse from data configs
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/d693dcd2b46c59d5478960d0fba52fdc1174763d/lm_eval/api/task.py#L618
        # Mock the lm_eval filter registry to use our filters
        # with patch.object(lm_eval.filters, "apply_filter_list") as mock_apply:

    #         # Set up the mock to simulate the filter application
    #         if "#### 42" in input_text:
    #             mock_apply.return_value = "42"  # pound filter matches
    #         elif "answer is" in input_text:
    #             mock_apply.return_value = input_text.split("answer is")[1].strip().lstrip("$:")
    #         elif "$" in input_text:
    #             mock_apply.return_value = input_text.split("$")[1].split()[0]
    #         else:
    #             mock_apply.return_value = expected_result  # For the "no match" case

    #         # Call the filter application
    #         from lm_eval.filters import apply_filter_list
    #         result = apply_filter_list(filter_config["filter_list"], input_text)

    #         # Check the result
    #         assert result == expected_result, f"Failed on input: {input_text}"

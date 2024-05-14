from unit_test_eval import generate
import pytest
from langsmith import unit
from langsmith import expect

# Basic assertion test
@unit
def test_generate():
    query = "How do I write a for loop with range 10 in Python?"
    output = generate(query)
    assert output == "for i in range(10):"

# Using langsmith's expect with .to_contain() method
@unit
def test_generate_2():
    query = "How do I make a list in python?"
    output = generate(query)
    expect(output).to_contain("[]")

# Using langsmith's expect with .embedding_distance() and .edit_distance() for fuzzy matching
@unit
def test_generate_3():
    query = "How do I write hello world in Python?"
    reference = 'print("Hello, World!")'
    output = generate(query)
    # Embedding Distance
    expect.embedding_distance(prediction=output, reference=reference).to_be_less_than(0.5)
    # Damerau-Levenshtein Edit Distance
    expect.edit_distance(prediction=output, reference=reference)
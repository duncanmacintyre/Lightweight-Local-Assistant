import os
import pytest
import asyncio
from lightweight_local_assistant import complete_plan_step

@pytest.mark.asyncio
async def test_complete_plan_step():
    # Setup
    os.makedirs(".gemini", exist_ok=True)
    plan_path = ".gemini/local_plan.md"
    plan_content = """# Plan
- [ ] Step 1
- [ ] Step 2
- [ ] Step 3
"""
    with open(plan_path, "w") as f:
        f.write(plan_content)
        
    # Test step 1
    res = await complete_plan_step(1)
    assert "Successfully marked step 1" in res
    with open(plan_path, "r") as f:
        content = f.read()
        assert "- [x] Step 1" in content
        assert "- [ ] Step 2" in content
        
    # Test step 2
    res = await complete_plan_step(2)
    assert "Successfully marked step 2" in res
    with open(plan_path, "r") as f:
        content = f.read()
        assert "- [x] Step 2" in content
        
    # Test step 4 (out of bounds)
    res = await complete_plan_step(4)
    assert "Error: Could not find step 4" in res
    
    # Test already completed
    res = await complete_plan_step(1)
    assert "already marked as complete" in res
    
    # Cleanup
    os.remove(plan_path)
    os.rmdir(".gemini")

if __name__ == "__main__":
    asyncio.run(test_complete_plan_step())
    print("Test passed!")

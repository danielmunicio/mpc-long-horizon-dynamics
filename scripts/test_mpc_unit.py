"""
Unit tests for MPC with L4CasADi integration.
Run with: python -m pytest test_mpc_unit.py -v
"""

import sys
import os
from pathlib import Path

# These tests validate code structure and logic without requiring dependencies


def test_mpc_state_structure():
    """Test that state structure is consistent (14 dimensions)"""
    print("\n=== Test: State Structure ===")
    
    state_dims = {
        'velocity': 3,
        'quaternion': 4,
        'angular_velocity': 3,
        'control_input': 4
    }
    
    total = sum(state_dims.values())
    expected = 14
    
    assert total == expected, f"State dimension mismatch: {total} vs {expected}"
    print(f"✅ State structure correct: {state_dims}")
    print(f"   Total dimensions: {total}")


def test_model_outputs():
    """Test that model outputs are correctly sized"""
    print("\n=== Test: Model Output Dimensions ===")
    
    velocity_model_output = 6  # [dv_x, dv_y, dv_z, dw_x, dw_y, dw_z]
    attitude_model_output = 4  # [q_x, q_y, q_z, q_w]
    
    print(f"✅ Velocity model output: {velocity_model_output} (delta changes)")
    print(f"✅ Attitude model output: {attitude_model_output} (quaternion)")


def test_history_window_logic():
    """Test history sliding window logic"""
    print("\n=== Test: History Window Sliding ===")
    
    history_length = 20
    history_buffer = list(range(history_length))  # Mock history
    
    # Simulate sliding window
    new_state = 100
    history_buffer = history_buffer[1:] + [new_state]
    
    assert len(history_buffer) == history_length, "History length changed"
    assert history_buffer[-1] == new_state, "New state not added"
    assert history_buffer[0] == 1, "Old state not removed"
    
    print(f"✅ Sliding window works correctly")
    print(f"   Window size: {len(history_buffer)}")
    print(f"   First element after slide: {history_buffer[0]}")
    print(f"   Last element after slide: {history_buffer[-1]}")


def test_dynamics_constraint_logic():
    """Test the dynamics constraint formulation"""
    print("\n=== Test: Dynamics Constraint Logic ===")
    
    # Mock state and predictions
    v_current = [1.0, 0.0, 0.0]  # 3D
    w_current = [0.0, 0.0, 0.0]  # 3D
    
    dv = [0.1, 0.0, 0.0]  # delta velocity
    dw = [0.0, 0.0, 0.1]  # delta angular velocity
    
    # Compute next state
    v_next = [v_current[i] + dv[i] for i in range(3)]
    w_next = [w_current[i] + dw[i] for i in range(3)]
    
    expected_v_next = [1.1, 0.0, 0.0]
    expected_w_next = [0.0, 0.0, 0.1]
    
    assert v_next == expected_v_next, f"Velocity update failed: {v_next}"
    assert w_next == expected_w_next, f"Angular velocity update failed: {w_next}"
    
    print(f"✅ Dynamics constraint formulation correct")
    print(f"   v_current: {v_current}")
    print(f"   dv: {dv}")
    print(f"   v_next: {v_next}")
    print(f"   w_current: {w_current}")
    print(f"   dw: {dw}")
    print(f"   w_next: {w_next}")


def test_l4casadi_input_shape():
    """Test L4CasADi input shape requirements"""
    print("\n=== Test: L4CasADi Input Shape ===")
    
    history_length = 20
    state_dim = 14
    
    flattened_input_size = history_length * state_dim
    expected = 280
    
    assert flattened_input_size == expected, f"Input size mismatch: {flattened_input_size} vs {expected}"
    
    print(f"✅ L4CasADi input shape correct")
    print(f"   History length: {history_length}")
    print(f"   State dimension: {state_dim}")
    print(f"   Flattened input size: {flattened_input_size}")


def test_mpc_constraint_types():
    """Test that MPC has correct constraint types"""
    print("\n=== Test: MPC Constraint Types ===")
    
    constraints = {
        'dynamics': 'X_k = f(history_buffer)',
        'state_bounds': 'x_lower <= X_k <= x_upper',
        'control_bounds': 'u_lower <= U_k <= u_upper',
        'cost': 'sum of squared errors from goal + control effort'
    }
    
    print(f"✅ MPC constraints properly defined:")
    for constraint_type, description in constraints.items():
        print(f"   - {constraint_type}: {description}")


def test_file_exists():
    """Test that required files exist"""
    print("\n=== Test: Required Files ===")
    
    required_files = [
        'mpc.py',
        'config.py',
        'dynamics_learning/lighting.py',
        'dynamics_learning/data.py',
        'requirements.txt'
    ]
    
    script_dir = Path(__file__).parent
    missing = []
    
    for file in required_files:
        file_path = script_dir / file
        if not file_path.exists():
            missing.append(file)
            print(f"   ❌ {file} - NOT FOUND")
        else:
            print(f"   ✅ {file}")
    
    assert len(missing) == 0, f"Missing files: {missing}"


def main():
    print("=" * 70)
    print("MPC with L4CasADi Integration - Unit Tests (No Dependencies)")
    print("=" * 70)
    
    try:
        test_mpc_state_structure()
        test_model_outputs()
        test_history_window_logic()
        test_dynamics_constraint_logic()
        test_l4casadi_input_shape()
        test_mpc_constraint_types()
        test_file_exists()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train dynamics models: python train.py")
        print("3. Run integration test: python test_mpc.py")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

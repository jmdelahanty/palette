#!/usr/bin/env python3
"""
TensorRT trtexec Diagnostics Script
Test and validate trtexec installation and command format.
"""

import subprocess
import argparse
from pathlib import Path
import tempfile
import os

def test_trtexec_installation(trtexec_path):
    """Test basic trtexec installation."""
    print(f"🔍 Testing trtexec installation at: {trtexec_path}")
    
    if not Path(trtexec_path).exists():
        print(f"❌ trtexec not found at {trtexec_path}")
        return False
    
    try:
        # Test version
        result = subprocess.run([trtexec_path, "--version"], 
                              capture_output=True, text=True, timeout=10)
        print(f"✅ trtexec version check:")
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"   {line}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing trtexec: {e}")
        return False

def test_basic_command_format(trtexec_path):
    """Test basic command format without actually building."""
    print(f"\n🧪 Testing basic command format...")
    
    # Test help command
    try:
        result = subprocess.run([trtexec_path, "--help"], 
                              capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ --help command works")
            
            # Check for workspace parameter format
            help_text = result.stdout
            if "--memPoolSize" in help_text:
                print("✅ Found --memPoolSize parameter (TensorRT 10.x format)")
                return "v10"
            elif "--workspace" in help_text:
                print("✅ Found --workspace parameter (older TensorRT format)")
                return "legacy"
            else:
                print("⚠️  Could not determine workspace parameter format")
                return "unknown"
        else:
            print(f"❌ --help command failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing help command: {e}")
        return None

def generate_test_onnx(output_path):
    """Generate a minimal test ONNX model."""
    print(f"📦 Generating minimal test ONNX model...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a very simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 3, padding=1)
                
            def forward(self, x):
                return self.conv(x)
        
        model = SimpleModel()
        model.eval()
        
        # Export to ONNX
        dummy_input = torch.randn(1, 3, 64, 64)
        torch.onnx.export(
            model, dummy_input, output_path,
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
        
        print(f"✅ Test ONNX model created: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating test ONNX: {e}")
        return False

def test_engine_building(trtexec_path, onnx_path, format_version):
    """Test actual engine building with correct command format."""
    print(f"\n🔧 Testing engine building...")
    
    engine_path = onnx_path.with_suffix('.engine')
    
    # Build command based on detected format
    if format_version == "v10":
        cmd = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--memPoolSize=workspace:1024MiB",
            "--fp16"
        ]
    elif format_version == "legacy":
        cmd = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--workspace=1024",
            "--fp16"
        ]
    else:
        # Try both formats
        print("⚠️  Unknown format, trying TensorRT 10.x format first...")
        cmd = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--memPoolSize=workspace:1024MiB",
            "--fp16"
        ]
    
    print(f"🚀 Test command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and engine_path.exists():
            print("✅ Test engine building successful!")
            print(f"   Engine created: {engine_path}")
            print(f"   Engine size: {engine_path.stat().st_size / 1024:.1f} KB")
            return True
        else:
            print(f"❌ Test engine building failed (return code: {result.returncode})")
            print("STDOUT:")
            print(result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:")
            print(result.stderr[-1000:])  # Last 1000 chars
            
            # If v10 format failed, try legacy format
            if format_version == "unknown":
                print("\n🔄 Trying legacy format...")
                cmd_legacy = [
                    trtexec_path,
                    f"--onnx={onnx_path}",
                    f"--saveEngine={engine_path}",
                    "--workspace=1024",
                    "--fp16"
                ]
                
                result2 = subprocess.run(cmd_legacy, capture_output=True, text=True, timeout=60)
                if result2.returncode == 0 and engine_path.exists():
                    print("✅ Legacy format worked!")
                    return True
                else:
                    print("❌ Legacy format also failed")
            
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out (this might be normal for complex models)")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Diagnose TensorRT trtexec issues")
    parser.add_argument("--trtexec-path", type=str, 
                       default="/usr/local/TensorRT-10.0.1.6/bin/trtexec",
                       help="Path to trtexec binary")
    parser.add_argument("--test-onnx", type=str,
                       help="Path to existing ONNX file to test with")
    
    args = parser.parse_args()
    
    print("🔧 TensorRT trtexec Diagnostics")
    print("=" * 50)
    
    # Test 1: Basic installation
    if not test_trtexec_installation(args.trtexec_path):
        print("\n❌ Basic installation test failed. Exiting.")
        return
    
    # Test 2: Command format detection
    format_version = test_basic_command_format(args.trtexec_path)
    if format_version is None:
        print("\n❌ Command format test failed. Exiting.")
        return
    
    # Test 3: Engine building
    if args.test_onnx:
        onnx_path = Path(args.test_onnx)
        if not onnx_path.exists():
            print(f"❌ ONNX file not found: {onnx_path}")
            return
    else:
        # Create temporary test ONNX
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = Path(temp_dir) / "test_model.onnx"
            if not generate_test_onnx(onnx_path):
                print("\n❌ Could not create test ONNX. Skipping engine building test.")
                return
            
            if not test_engine_building(args.trtexec_path, onnx_path, format_version):
                print("\n❌ Engine building test failed")
                print("\n💡 Recommendations:")
                print("   1. Check your TensorRT installation")
                print("   2. Verify CUDA and cuDNN are properly installed")
                print("   3. Try a different precision (fp32 instead of fp16)")
                print("   4. Check if your GPU supports the requested precision")
                return
    
    print("\n🎉 All diagnostics passed!")
    print(f"💡 Your trtexec installation appears to be working correctly")
    print(f"🔧 Detected format: {format_version}")
    
    if format_version == "v10":
        print("\n📋 For your models, use this command format:")
        print("   --memPoolSize=workspace:4096MiB")
    elif format_version == "legacy":
        print("\n📋 For your models, use this command format:")
        print("   --workspace=4096")

if __name__ == "__main__":
    main()
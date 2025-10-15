#!/usr/bin/env python3
"""
M3芯片兼容性检查脚本
运行此脚本来验证您的系统是否支持MPS加速
"""

import sys
import platform

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("✅ Python版本兼容")
        return True
    else:
        print("❌ Python版本不兼容，需要Python 3.8或更高版本")
        return False

def check_system():
    """检查系统信息"""
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("✅ 检测到Apple Silicon Mac")
        return True
    else:
        print("⚠️  未检测到Apple Silicon Mac，MPS可能不可用")
        return False

def check_torch():
    """检查PyTorch和MPS支持"""
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        # 检查MPS支持
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        
        print(f"MPS可用: {mps_available}")
        print(f"MPS已构建: {mps_built}")
        
        if mps_available and mps_built:
            print("✅ MPS加速可用")
            
            # 测试MPS设备
            try:
                device = torch.device("mps")
                test_tensor = torch.randn(10, 10).to(device)
                print("✅ MPS设备测试成功")
                return True
            except Exception as e:
                print(f"❌ MPS设备测试失败: {e}")
                return False
        else:
            print("❌ MPS不可用")
            return False
            
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def check_other_dependencies():
    """检查其他依赖"""
    dependencies = [
        "pytorch_lightning",
        "transformers", 
        "datasets",
        "scikit_learn",
        "numpy"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} 已安装")
        except ImportError:
            print(f"❌ {dep} 未安装")
            missing.append(dep)
    
    return len(missing) == 0, missing

def main():
    """主检查函数"""
    print("=" * 50)
    print("M3芯片兼容性检查")
    print("=" * 50)
    
    checks = []
    
    # 检查Python版本
    checks.append(check_python_version())
    print()
    
    # 检查系统
    checks.append(check_system())
    print()
    
    # 检查PyTorch
    checks.append(check_torch())
    print()
    
    # 检查其他依赖
    deps_ok, missing = check_other_dependencies()
    checks.append(deps_ok)
    print()
    
    # 总结
    print("=" * 50)
    print("检查结果总结:")
    print("=" * 50)
    
    if all(checks):
        print("🎉 所有检查通过！您的系统完全兼容M3芯片加速")
        print("\n可以运行以下命令开始训练:")
        print("python train.py")
    else:
        print("⚠️  部分检查未通过，请根据上述提示解决问题")
        if missing:
            print(f"\n缺少的依赖: {', '.join(missing)}")
            print("请运行: pip install -r requirements_m3.txt")

if __name__ == "__main__":
    main()

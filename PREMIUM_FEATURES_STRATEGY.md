# Premium Features Strategy for 360FrameTools

## Overview
Maintain open-source core while monetizing advanced features on Gumroad.

---

## Architecture: Feature Flags

### Option 1: Build-Time Flags (Recommended)
```python
# src/config/features.py
class FeatureFlags:
    """Feature availability based on build type"""
    
    # Determined at build time (set by build script)
    IS_PREMIUM = False  # Set to True for Gumroad builds
    
    # Public features (always available)
    FRAME_EXTRACTION = True
    E2P_TRANSFORM = True
    E2C_TRANSFORM = True
    YOLO_MASKING_BASIC = True  # Persons only
    
    # Premium features (Gumroad exclusive)
    DIRECT_EQUIRECT_MASKING = IS_PREMIUM
    SAM3_INTERACTIVE_MASKING = IS_PREMIUM
    YOLO_EXTENDED_CATEGORIES = IS_PREMIUM  # Animals, objects
    ADVANCED_PRESETS = IS_PREMIUM  # >8 cameras, custom FOV curves
    GPU_BATCH_OPTIMIZATION = IS_PREMIUM  # Multi-GPU, advanced queuing
    PRIORITY_SUPPORT = IS_PREMIUM
```

### Option 2: License Key Validation
```python
# src/config/license.py
import hashlib
from pathlib import Path

class LicenseManager:
    """Validates Gumroad license keys"""
    
    def __init__(self):
        self.license_file = Path.home() / ".360toolkit" / "license.key"
        self._is_valid = self._check_license()
    
    def _check_license(self) -> bool:
        """Validate license key against Gumroad email hash"""
        if not self.license_file.exists():
            return False
        
        try:
            key = self.license_file.read_text().strip()
            # Validate against your secret salt + Gumroad email
            # (Implement your validation logic)
            return self._validate_key(key)
        except:
            return False
    
    @property
    def is_premium(self) -> bool:
        return self._is_valid
```

---

## Repository Structure

### Two-Repository Approach (Most Secure)
```
PUBLIC REPO (GitHub)
├─ 360FrameTools-Community/
│  ├─ src/
│  │  ├─ extraction/        # Public SDK integration
│  │  ├─ transforms/        # E2P/E2C (public)
│  │  ├─ masking/           
│  │  │  └─ basic_masker.py # YOLO persons only
│  │  └─ ui/                # Basic UI
│  ├─ LICENSE               # GPL-3.0
│  └─ README.md             # "Premium features at gumroad.com/..."

PRIVATE REPO (Your machine + Gumroad builds)
├─ 360FrameTools-Premium/
│  ├─ src/
│  │  ├─ masking/
│  │  │  ├─ sam3_masker.py       # SAM3 integration
│  │  │  └─ extended_masker.py   # Animals, objects
│  │  ├─ features/
│  │  │  ├─ direct_masking.py    # Equirect/fisheye masking
│  │  │  └─ advanced_presets.py  # Custom camera configs
│  │  └─ ui/
│  │     └─ premium_ui.py        # Interactive SAM UI
│  ├─ LICENSE                    # Proprietary
│  └─ build_gumroad.py           # Build script with premium code
```

### Single-Repository with Private Submodule (Easier Management)
```
360FrameTools/ (PUBLIC)
├─ src/
│  ├─ public/          # All public code
│  └─ premium/         # Git submodule (PRIVATE repo)
│     ├─ sam3_masker.py
│     ├─ direct_masking.py
│     └─ ...
├─ .gitignore          # Ignore premium/ in public builds
└─ build_scripts/
   ├─ build_public.py  # Excludes premium/
   └─ build_premium.py # Includes premium/
```

---

## UI Integration

### Premium Feature Teaser UI
```python
# src/ui/premium_widgets.py
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton
from PyQt6.QtCore import Qt

class PremiumFeatureWidget(QWidget):
    """Shows locked feature with upgrade prompt"""
    
    def __init__(self, feature_name: str, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        
        # Feature preview (grayed out)
        preview = QLabel(f"🔒 {feature_name}")
        preview.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(preview)
        
        # Description
        desc = QLabel("Available in Premium Edition")
        desc.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(desc)
        
        # Upgrade button
        btn = QPushButton("Unlock on Gumroad →")
        btn.clicked.connect(self.open_gumroad)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #FF90E8;
                color: white;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #FF70D8;
            }
        """)
        layout.addWidget(btn)
        
        self.setLayout(layout)
    
    def open_gumroad(self):
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://gumroad.com/your-product"))
```

### Conditional UI Loading
```python
# src/ui/main_window.py
from src.config.features import FeatureFlags

def create_masking_tab(self):
    """Create Stage 3 masking configuration"""
    
    # Basic YOLO (always available)
    layout.addWidget(self.create_yolo_basic_controls())
    
    # Premium features
    if FeatureFlags.SAM3_INTERACTIVE_MASKING:
        layout.addWidget(self.create_sam3_controls())
    else:
        layout.addWidget(PremiumFeatureWidget("SAM3 Interactive Masking"))
    
    if FeatureFlags.DIRECT_EQUIRECT_MASKING:
        layout.addWidget(self.create_direct_masking_controls())
    else:
        layout.addWidget(PremiumFeatureWidget("Direct Equirectangular Masking"))
```

---

## Build Process

### Automated Build Scripts
```python
# scripts/build_public.py
"""Build community edition for GitHub releases"""
import shutil
from pathlib import Path

def build_public():
    # Set feature flags
    features_path = Path("src/config/features.py")
    content = features_path.read_text()
    content = content.replace("IS_PREMIUM = False", "IS_PREMIUM = False")
    features_path.write_text(content)
    
    # Remove premium modules
    premium_dir = Path("src/premium")
    if premium_dir.exists():
        shutil.rmtree(premium_dir)
    
    # Build with PyInstaller
    subprocess.run([
        "pyinstaller",
        "--name=360FrameTools-Community",
        "360ToolkitCommunity.spec"
    ])
    
    print("✅ Public build ready for GitHub release")

# scripts/build_premium.py
"""Build Gumroad edition with all features"""
def build_premium():
    # Set premium flag
    features_path = Path("src/config/features.py")
    content = features_path.read_text()
    content = content.replace("IS_PREMIUM = False", "IS_PREMIUM = True")
    features_path.write_text(content)
    
    # Include premium modules
    # (already present in local dev environment)
    
    # Build with PyInstaller
    subprocess.run([
        "pyinstaller",
        "--name=360FrameTools-Premium",
        "360ToolkitPremium.spec"
    ])
    
    print("✅ Premium build ready for Gumroad")
```

---

## Licensing Strategy

### Dual Licensing
```
PUBLIC (GitHub):
- License: GPL-3.0 (or MIT if you prefer permissive)
- Anyone can use, modify, distribute
- Cannot use for closed-source commercial products

PREMIUM (Gumroad):
- License: Proprietary + Commercial Use Rights
- Single-user or multi-seat options
- Includes premium features + priority support
- Allowed for commercial projects
```

### License File (Public Repo)
```markdown
# LICENSE

## 360FrameTools Community Edition

This software is licensed under the GNU General Public License v3.0 (GPL-3.0).

Premium features are available exclusively through Gumroad under a separate 
proprietary license. The premium edition includes:
- SAM3 Interactive Masking
- Direct Equirectangular/Fisheye Masking
- Extended YOLO Categories (Animals, Objects)
- Advanced Camera Presets
- GPU Batch Optimization
- Priority Support & Updates

Purchase: https://gumroad.com/your-product-link
```

---

## Pricing Tiers (Suggested)

| Tier | Price | Features |
|------|-------|----------|
| **Community** | Free | Frame extraction, E2P/E2C, YOLO persons, 8 presets |
| **Professional** | $29 | + Direct masking, animals/objects, 20 presets |
| **Studio** | $99 | + SAM3 interactive, GPU optimization, unlimited presets, priority support |
| **Commercial** | $299 | + Multi-seat (5 users), commercial license, phone support |

---

## Anti-Piracy Measures (Optional)

### 1. License Key Validation
- Generate unique keys per Gumroad purchase
- Validate on app startup (offline check)
- Store hashed key in user home directory

### 2. Gumroad API Integration
```python
# Validate license via Gumroad API
import requests

def verify_license(license_key: str) -> bool:
    response = requests.post(
        "https://api.gumroad.com/v2/licenses/verify",
        data={
            "product_id": "YOUR_PRODUCT_ID",
            "license_key": license_key
        }
    )
    return response.json().get("success", False)
```

### 3. Obfuscation (PyArmor)
- Obfuscate premium modules only
- Makes reverse engineering harder
- `pip install pyarmor`

---

## Marketing Strategy

### GitHub README.md
```markdown
# 360FrameTools Community Edition

Free, open-source tool for Insta360 frame extraction and photogrammetry.

## Features
✅ Frame extraction from .INSV files
✅ Perspective splitting (E2P, E2C)
✅ Basic YOLO masking (persons)
✅ 8 camera presets

## Premium Edition 🚀
Unlock advanced features:
- 🎯 SAM3 interactive masking (click to segment any object)
- 🌐 Direct equirectangular/fisheye masking
- 🐾 Extended categories (animals, objects)
- ⚡ GPU batch optimization (3-5× faster)
- 🎨 Unlimited custom camera presets
- 💬 Priority support & updates

[**Get Premium on Gumroad** →](https://gumroad.com/your-link)
```

---

## Implementation Checklist

### Phase 1: Feature Flag System
- [ ] Create `src/config/features.py` with `FeatureFlags` class
- [ ] Update UI to check flags before showing premium features
- [ ] Add `PremiumFeatureWidget` for locked features
- [ ] Test public build with all premium features disabled

### Phase 2: Premium Features Development
- [ ] Implement direct equirect/fisheye masking pipeline
- [ ] Integrate SAM3 model (separate branch/repo)
- [ ] Create interactive masking UI
- [ ] Add extended YOLO categories (animals, objects)

### Phase 3: Build System
- [ ] Create `build_public.py` and `build_premium.py` scripts
- [ ] Set up separate PyInstaller specs for each version
- [ ] Test both builds on clean Windows machine
- [ ] Verify premium features work only in premium build

### Phase 4: Repository Management
- [ ] Split code into public + private submodule (or separate repos)
- [ ] Update `.gitignore` to exclude premium code from public commits
- [ ] Set up GitHub Actions for automated public builds
- [ ] Document contribution guidelines (public repo only)

### Phase 5: Gumroad Setup
- [ ] Create Gumroad product page with screenshots
- [ ] Set up license key generation (optional)
- [ ] Write installation guide for premium buyers
- [ ] Set up automated delivery (zip file)

---

## Conclusion

**Recommended Approach**:
1. **Keep current YOLO system** for public version (persons only)
2. **Add SAM3 as premium feature** (interactive masking mode)
3. **Add direct equirect masking as premium** (skip transform)
4. **Use build-time feature flags** (simplest to maintain)
5. **Two-build system**: Public (GitHub) + Premium (Gumroad)

This strategy:
- ✅ Maintains goodwill with open-source community
- ✅ Monetizes advanced features fairly
- ✅ Provides clear upgrade path
- ✅ Easy to maintain (one codebase with flags)
- ✅ Protects your intellectual property

---

**Need Help Implementing?**
Let me know which features to prioritize and I'll start coding!

# 🚀 GitHub Codespaces Setup with Claude Code Pro

## Complete Guide for Cloud Development with AI Assistant

---

## 🎯 What You Get

- ✅ **Full VSCode in Browser** - No local setup needed
- ✅ **Claude Code Pro AI Assistant** - Unlimited usage with your Pro subscription
- ✅ **4-8GB RAM** - TensorFlow runs smoothly
- ✅ **Full Debugging** - Breakpoints, variable inspection, call stack
- ✅ **Public Dashboard URL** - Share your work instantly
- ✅ **Git Integration** - Commit and push directly
- ✅ **60 Hours FREE/Month** - Generous free tier

---

## 🚀 Quick Start (3 Minutes)

### Step 1: Open Codespace (1 Click)

**Click this button:**

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/kishormuthal/IoT-Predictive-Maintenance-System)

**Or manually:**
1. Go to: https://github.com/kishormuthal/IoT-Predictive-Maintenance-System
2. Click green **"Code"** button
3. Click **"Codespaces"** tab
4. Click **"Create codespace on main"**

### Step 2: Wait for Setup (2-3 Minutes)

Codespace automatically:
- ✅ Creates cloud VM
- ✅ Installs Python 3.11
- ✅ Installs all dependencies (`pip install -r requirements.txt`)
- ✅ Installs Claude Code extension
- ✅ Configures port forwarding

**You'll see:**
```
Setting up your codespace...
Running postCreateCommand...
Installing dependencies from requirements.txt...
✓ Setup complete!
```

### Step 3: Authenticate Claude Code (First Time Only)

When Codespace opens:

1. **Claude Code extension shows notification:**
   ```
   "Sign in to use Claude Code"
   [Sign In]
   ```

2. **Click "Sign In"** button

3. **Browser opens for OAuth authentication:**
   - Login with your Anthropic account
   - The one with Claude Code Pro subscription
   - Authorize the connection

4. **Return to Codespace:**
   - Authentication complete!
   - Claude Code icon active in sidebar
   - Ready to use with unlimited Pro access

**✅ No API key needed! Uses your Pro subscription automatically.**

### Step 4: Run Dashboard

In the terminal (bottom of VSCode):

```bash
# Quick validation
python quick_start.py

# Start dashboard
python start_dashboard.py
```

**Expected output:**
```
============================================================
IoT PREDICTIVE MAINTENANCE DASHBOARD
Clean Launch with All Trained Models
============================================================
[STATUS] All 12 forecasting models trained and ready
[STATUS] NASA SMAP/MSL data loaded for 12 sensors
[STATUS] Anti-hanging architecture enabled
------------------------------------------------------------
[INFO] Creating UNIFIED dashboard application...
[INFO] ALL features from src/ enabled - ZERO compromise
[URL] Dashboard starting at: http://127.0.0.1:8050
```

### Step 5: Access Dashboard

**Automatic (Recommended):**
- VSCode shows notification: **"Your application is available on port 8050"**
- Click **"Open in Browser"**
- Dashboard opens in new tab

**Manual:**
1. Click **"Ports"** tab at bottom
2. Find port **8050** (labeled "IoT Dashboard")
3. Click globe icon 🌐
4. Dashboard opens

---

## 🤖 Using Claude Code AI Assistant

### Features Available with Pro Subscription:

#### 1. **Chat with Claude**
- Click **Claude icon** in left sidebar
- Ask questions about code:
  - "Explain how unified_dashboard.py works"
  - "Why is TensorFlow loading slowly?"
  - "How do I add a new tab to the dashboard?"

#### 2. **Inline Code Generation**
- Select code
- Right-click → **"Ask Claude"**
- Type what you want:
  - "Add error handling here"
  - "Optimize this function"
  - "Add docstring"

#### 3. **Debug with AI**
- Click Claude icon
- Ask: "Debug dashboard startup issue"
- Claude analyzes code and suggests fixes
- Apply fixes with 1 click

#### 4. **Generate Code**
- Press `Ctrl+Shift+P`
- Type: **"Claude: Generate Code"**
- Describe what you want
- Claude writes the code

#### 5. **Explain Complex Code**
- Select complex section
- Right-click → **"Ask Claude"**
- Ask: "Explain this code"
- Claude provides detailed explanation

#### 6. **Refactor & Optimize**
- Select code
- Ask Claude: "Refactor this to be more efficient"
- Review suggestions
- Apply changes

#### 7. **Write Tests**
- Open function
- Ask Claude: "Write pytest tests for this function"
- Tests generated automatically

#### 8. **Git Commit Messages**
- Stage changes
- Ask Claude: "Generate commit message"
- Meaningful message created

---

## 🛠️ Development Workflow

### Typical Session:

```
1. Open Codespace (1 click)
   ↓
2. Codespace loads (2-3 min first time, 30 sec after)
   ↓
3. Edit code with VSCode
   ↓
4. Ask Claude Code for help as needed
   ↓
5. Test: python start_dashboard.py
   ↓
6. Debug with breakpoints if needed
   ↓
7. Commit changes
   ↓
8. Push to GitHub (built-in Git)
   ↓
9. Close Codespace (stops billing hours)
```

### Example: Fixing TensorFlow Loading Issue

**Problem:** Dashboard takes 3 minutes to start

**Solution with Claude Code:**

1. **Ask Claude:**
   ```
   "Dashboard takes 3 minutes to start. TensorFlow seems to be
   the issue. How can I make it load faster?"
   ```

2. **Claude suggests:**
   ```python
   # Use lazy imports for TensorFlow
   def _get_model(self, sensor_id: str):
       import tensorflow as tf  # Import only when needed
       # Rest of code...
   ```

3. **Apply fix** (1 click)

4. **Test:**
   ```bash
   python start_dashboard.py
   # Now starts in 30 seconds!
   ```

5. **Commit:**
   - Ask Claude: "Generate commit message for this change"
   - Claude: "feat: Optimize TensorFlow loading with lazy imports"
   - Commit and push

---

## 🐛 Debugging in Codespaces

### Using VSCode Debugger:

#### Set Breakpoint:
1. Open `src/presentation/dashboard/unified_dashboard.py`
2. Click left of line number (red dot appears)
3. Common breakpoints:
   - Line 320: `def _initialize_services_safely()`
   - Line 450: `def _get_tab_content()`
   - Line 550: Service initialization

#### Start Debugging:
1. Press `F5` or Run → Start Debugging
2. Select: **"Python File"**
3. Dashboard starts in debug mode
4. Execution pauses at breakpoints

#### Debug Controls:
- `F10` - Step over
- `F11` - Step into
- `Shift+F11` - Step out
- `F5` - Continue
- `Shift+F5` - Stop

#### Inspect Variables:
- Hover over variable
- Check "Variables" panel
- Add to "Watch" panel
- Evaluate in Debug Console

### Using Claude Code for Debugging:

#### Analyze Errors:
```
1. Copy error message
2. Ask Claude: "I got this error: [paste error]. What's wrong?"
3. Claude explains issue and suggests fix
4. Apply fix
```

#### Understand Complex Behavior:
```
1. Select problematic code section
2. Ask Claude: "Why isn't this working as expected?"
3. Claude analyzes code flow
4. Points out issue
```

---

## 📂 File Navigation

### Quick Access:

- **Ctrl+P** - Quick file open
- **Ctrl+Shift+F** - Search in files
- **Ctrl+Shift+E** - Explorer sidebar
- **Ctrl+`** - Toggle terminal

### Key Files:

```
├── start_dashboard.py              # Main launcher
├── src/presentation/dashboard/
│   └── unified_dashboard.py        # Main dashboard (EDIT THIS)
├── .devcontainer/
│   └── devcontainer.json          # Codespaces config
├── requirements.txt               # Dependencies
└── docs/
    └── UNIFIED_DASHBOARD.md       # Dashboard docs
```

---

## 🔧 Common Tasks

### Add New Feature:
```
1. Ask Claude: "How do I add a new tab to the dashboard?"
2. Claude explains process and provides code
3. Apply changes
4. Test: python start_dashboard.py
5. Commit and push
```

### Fix Bug:
```
1. Find bug in code
2. Set breakpoint
3. Press F5 to debug
4. Or ask Claude: "This isn't working: [describe issue]"
5. Apply fix
6. Test
```

### Optimize Performance:
```
1. Select slow code section
2. Ask Claude: "How can I optimize this?"
3. Claude suggests improvements
4. Apply and benchmark
```

### Write Tests:
```
1. Open function to test
2. Ask Claude: "Write pytest tests for this function"
3. Tests generated
4. Run: pytest tests/
```

### Generate Documentation:
```
1. Select undocumented code
2. Ask Claude: "Add comprehensive docstrings"
3. Documentation added
4. Commit
```

---

## 💡 Tips & Best Practices

### Codespace Management:

✅ **Stop when done:** Click Codespace name → "Stop codespace"
- Saves your 60 free hours
- Work persists when stopped
- Restart takes 30 seconds

✅ **Name your Codespaces:**
- Helps identify multiple workspaces
- Settings → Codespace name

✅ **Use prebuilds:**
- Faster startup after first time
- Dependencies cached

### Claude Code Best Practices:

✅ **Be specific:** "Optimize TensorFlow loading" > "Make it faster"

✅ **Provide context:** Include error messages, file names, line numbers

✅ **Iterate:** Ask follow-up questions to refine solutions

✅ **Review code:** Claude generates good code, but always review

✅ **Learn:** Ask "Why?" to understand solutions

### Development Best Practices:

✅ **Commit often:** Small, focused commits

✅ **Test locally:** Run dashboard before pushing

✅ **Use branches:** Feature branches for new work

✅ **Ask Claude for help:** Don't struggle alone!

---

## 🚨 Troubleshooting

### Issue: Claude Code requires Node.js 18+

**Symptom:** "Claude Code requires Node.js version 18 or higher to be installed"

**Solution for NEW Codespaces (Automatic):**
- Node.js 20 is now configured in `.devcontainer/devcontainer.json`
- Just create a new Codespace and it will install automatically

**Solution for EXISTING Codespaces (Manual Install):**

**Option 1: Rebuild Codespace (Recommended)**
1. Press `Ctrl+Shift+P`
2. Type: **"Codespaces: Rebuild Container"**
3. Wait for rebuild (installs Node.js automatically)
4. Claude Code will work after rebuild

**Option 2: Install Node.js Manually**
```bash
# Install Node.js 20 using nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20
nvm alias default 20

# Verify installation
node --version  # Should show v20.x.x

# Reload VSCode window
# Press Ctrl+Shift+P → "Developer: Reload Window"
```

**Option 3: Create Fresh Codespace**
1. Stop current Codespace
2. Delete it (if you've committed your changes)
3. Create new Codespace (Node.js 20 auto-installs)

---

### Issue: Claude Code not showing

**Symptom:** No Claude icon in sidebar

**Solution:**
1. Press `Ctrl+Shift+P`
2. Type: **"Developer: Reload Window"**
3. Wait for extensions to load
4. Claude icon should appear

**Still not working?**
1. Check extensions: Click Extensions icon
2. Find "Claude Code"
3. Click "Reload" or "Enable"
4. If error about Node.js, see section above

---

### Issue: Authentication failed

**Symptom:** "Sign in to use Claude Code" keeps appearing

**Solution:**
1. Click **"Sign In"** again
2. Make sure pop-ups are not blocked
3. Use same Anthropic account with Pro subscription
4. Check subscription is active: https://console.anthropic.com/settings/billing

---

### Issue: Dashboard won't start

**Symptom:** Hangs at "Loading TensorFlow"

**Solution with Claude Code:**
1. Ask Claude: "Dashboard hangs loading TensorFlow. How to fix?"
2. Claude suggests: Lazy imports or timeout increases
3. Apply suggested fix
4. Test again

**Manual debugging:**
1. Set breakpoint at line 320 in unified_dashboard.py
2. Press F5 to debug
3. Step through to find blocking code
4. Ask Claude for optimization suggestions

---

### Issue: Port 8050 not forwarding

**Symptom:** Can't access dashboard URL

**Solution:**
1. Check "Ports" tab at bottom
2. Verify port 8050 is listed
3. Visibility should be "Public"
4. Click globe icon to open
5. If not listed, manually forward:
   - Ports tab → "Add Port"
   - Enter: 8050

---

### Issue: Out of Codespace hours

**Symptom:** "You've used your free hours"

**Solution:**
- Free tier: 60 hours/month
- Check usage: https://github.com/settings/billing
- Upgrade: $0.18/hour or GitHub Pro ($4/month for 90 hours)
- Stop Codespaces when not using to save hours

---

### Issue: Changes not saving

**Symptom:** Edits disappear after closing

**Solution:**
1. Codespaces auto-saves (usually)
2. Manual save: `Ctrl+S`
3. Commit changes:
   ```bash
   git add .
   git commit -m "Your changes"
   git push
   ```
4. Check Git status: `git status`

---

## 📊 Monitoring Usage

### Check Codespace Hours:
```
1. Go to: https://github.com/settings/billing
2. Scroll to "Codespaces"
3. See hours used this month
```

### Optimize Usage:
- **Stop Codespaces when done** (doesn't count toward hours)
- **Use prebuilds** (faster startup)
- **One Codespace at a time** (don't run multiple)
- **Delete unused Codespaces**

---

## 🎓 Learning Resources

### Claude Code Documentation:
- Official docs: https://docs.claude.com/en/docs/claude-code/overview
- Best practices: https://www.anthropic.com/engineering/claude-code-best-practices

### GitHub Codespaces:
- Official docs: https://docs.github.com/en/codespaces
- Quickstart: https://docs.github.com/en/codespaces/getting-started

### VSCode in Browser:
- VSCode web docs: https://code.visualstudio.com/docs/editor/vscode-web
- Keyboard shortcuts: https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf

---

## 🏆 Success Checklist

After following this guide, you should have:

- ✅ Codespace opened and running
- ✅ Claude Code authenticated with Pro subscription
- ✅ Dashboard running on port 8050
- ✅ Can access dashboard via public URL
- ✅ Can edit code with VSCode
- ✅ Can debug with breakpoints
- ✅ Can use Claude Code AI for help
- ✅ Can commit and push changes
- ✅ Know how to stop Codespace to save hours

---

## 💰 Cost Breakdown

### Free (60 hours/month):
- ✅ GitHub Codespaces: First 60 hours FREE
- ✅ Claude Code Pro: $20/month (unlimited in Codespaces)
- ✅ Total: $20/month

### If you need more:
- GitHub Codespaces extra hours: $0.18/hour
- Or GitHub Pro: $4/month (includes 90 Codespace hours)

**For 90 hours/month of development: $24/month total** ✅

---

## 🚀 Next Steps

1. **Now:** Open Codespace and test dashboard
2. **Today:** Explore Claude Code features
3. **This week:** Start using for development
4. **Ongoing:** Build and improve your IoT system!

---

## 📞 Getting Help

### Ask Claude Code:
- **In Codespace:** Click Claude icon → Ask your question
- Claude knows this entire codebase
- Can help with any issue

### GitHub Issues:
- **Report bugs:** https://github.com/kishormuthal/IoT-Predictive-Maintenance-System/issues
- Include: Error message, steps to reproduce, screenshots

### Documentation:
- **This guide:** `docs/CODESPACES_SETUP.md`
- **Unified Dashboard:** `docs/UNIFIED_DASHBOARD.md`
- **Main README:** `README.md`

---

**✅ You're all set! Happy coding with Codespaces + Claude Code Pro! 🎉**
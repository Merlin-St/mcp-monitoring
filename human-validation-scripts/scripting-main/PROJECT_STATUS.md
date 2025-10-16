# Gorilla Survey Builder - Project Documentation

**Last Updated**: Session 9 - Dual Study Support (Servers + Tools) 🔧

---

## ⚠️ KNOWN ISSUES - FIX TOMORROW

### Tools Study (main_tools.ts) - Tutorial Practice Pages Buggy
**Issue**: Pages 8/9/10 (tutorial feedback for practice questions) have validation/display errors

**Symptoms**:
- Page 8 (func_main feedback): May show "Please answer the question" alert even when answer was clicked
- Page 6/8/10 (feedback pages): May display "undefined" for user's answer
- Practice question responses not properly saved to `serverResponses['tutorial_tool_2']`

**Root Cause**:
- Individual question pages (`onet_l1`, `func_main`, `func_sub`) use different rendering logic than combined pages
- `validatePage()` function tries to validate practice pages but answer may not be stored correctly
- `showTutorialFeedbackPage()` tries to retrieve answer from `serverResponses[toolIndex]` but may be undefined

**Attempted Fixes** (not fully working):
1. ✅ Modified `showTutorialFeedbackPage()` to support `toolIndex` (line 1578 in generate_main_ts.py)
2. ✅ Added initialization of `serverResponses[idx]` in validation (line 2390 in generate_main_ts.py)
3. ✅ Made practice pages allow skipping validation (line 2382 in generate_main_ts.py)
4. ❌ Still buggy - needs further investigation

**TODO Tomorrow**:
- Debug where individual question page responses are being stored
- Check if `page.question` field is being properly used as key in `serverResponses`
- Verify `showFuncMainPage()`, `showONetL1Page()`, etc. functions exist and work correctly
- Consider simplifying tutorial to use combined pages instead of individual question pages
- Test thoroughly in browser with console.log to trace response storage

**Workaround**: Servers study (`main_servers.ts`) works fine - tutorials use same pattern but with `serverIndex`

---

## Table of Contents
1. [Current Status](#current-status)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Working Files](#working-files)
5. [Implementation Details](#implementation-details)
6. [Complete Session History](#complete-session-history)
7. [Testing & Validation](#testing--validation)
8. [Production Deployment](#production-deployment)

---

## Current Status

**Production Files**:
- `main_servers.ts` (29 pages, 10 servers, seed 42) - ✅ WORKING
- `main_tools.ts` (21 pages, 10 tools, seed 42) - ⚠️ TUTORIAL BUGGY (pages 8/9/10)

**Features**:
- Survey starts directly with Instructions page (no consent)
- Simplified O*NET classification: **Only Level 1** by default
- Tutorial 1: Pre-answered example - shows correct answers
- Tutorial 2: Practice with instant feedback - 3 questions (Q1.1, Q2.1, Q2.2)
- Servers study: Q0 + Q1.1 + Q2.1 + Q2.2 + Q3-Q5 = **7 questions per server**
- Tools study: Q1.1 + Q2.1 + Q2.2 = **3 questions per tool** (no Q0, no Q3-Q5)

**Command Line Options**:
- `--type servers` or `--type tools` - Choose study type
- `--servers NUM` - Number of items to include
- `--seed NUM` - Random seed for reproducibility
- `--all-onet-levels` - Enable all 3 O*NET levels (L1, L2, Task) - servers only

---

## Quick Start

### Generate and Test a Study

```bash
# Activate environment
source ~/mcp-monitoring/.venv/bin/activate

# Generate test survey with tutorials (2 servers for quick testing)
python3 generate_main_ts.py --servers 2 --output test_with_tutorials.ts --seed 42

# Generate larger survey for production
python3 generate_main_ts.py --servers 50 --output production_survey.ts --seed 999

# Start local server (if not running)
python3 -m http.server 8081

# Test in browser (increment ?v=2, ?v=3 for cache-busting)
# URL: http://localhost:8081/scripting-utils/test_dynamic.html?file=test_with_tutorials.ts&v=1
```

### Command Line Options

```bash
python3 generate_main_ts.py --help

Options:
  --servers NUM          Number of servers to include (default: 5)
  --data PATH           Path to server data JSON (default: ../../data/initial/data_unified_filtered.json)
  --questions PATH      Path to questions CSV (default: questions_config.csv)
  --output FILENAME     Output TypeScript filename (default: main.ts)
  --seed NUM            Random seed for reproducible server selection
```

---

## Architecture

### Why Generator-Based?

**Previous Approach (V1/V2 Manual)**:
- Manually edit TypeScript files
- Copy-paste server data
- Error-prone for large studies

**Current Approach (Generator)**:
- Python script generates TypeScript from data sources
- Loads servers from `data/initial/data_unified_filtered.json`
- Loads questions from `questions_config.csv`
- Single command generates complete study
- Reproducible with `--seed` parameter

### Data Flow

```
Server Data (JSON) + Questions (CSV)
           ↓
  generate_main_ts.py
           ↓
   Generated TypeScript (.ts)
           ↓
  test_dynamic.html (local testing)
           ↓
   Gorilla Experiment (production)
```

### Study Pattern

Based on **Gorilla Example 2 Pattern** (single-file, no templates):
- All logic in one TypeScript file
- HTML generated with jQuery
- Simple navigation with page index
- No complex imports or interfaces

## What Was Completed

### Session 1: Instruction Page
- Created `generate_instruction_page_content()` with O*NET and Functionality classification guide
- Fixed Unicode arrow issues
- Successfully renders after consent page

### Session 2: Generator Fix & Q0 Implementation
- **Fixed Critical Bug**: Missing `content:` property at line 605
- **Implemented Q0**: First question for each server with 20-second timer, textarea validation
- Generated `test_with_q0.ts` for testing

### Session 3: Documentation Verification
- Synchronized all documentation files
- Confirmed file cleanup (obsolete files deleted)

### Session 4: Tutorial Implementation ✅
**FULLY IMPLEMENTED AND BROWSER-TESTED**

Added 3 tutorial examples with complete TypeScript handlers:
1. **Tutorial 1 (asher-mcp)**: Pre-answered example with green answer blocks
2. **Tutorial 2 (base-mcp)**: Practice example with blue "Practice Mode" banner
3. **Tutorial 3 (DesktopCommanderMCP)**: Practice example

**Changes Made**:
- Line 654: Inserted `generate_tutorial_examples()` call
- Lines 854-859: Added tutorial handlers in `showPage()`
- Lines 1017-1071: Added `showTutorialPreansweredPage()` and `showTutorialPracticePage()` functions
- Lines 1303-1306: Added tutorial validation (no validation required)
- Line 1496: Updated page count to include 7 tutorial pages

**Browser Validation**: 9 screenshots confirm all features work correctly

### Session 5: Production Fixes ✅
**CRITICAL BUGS FIXED - SURVEY NOW PRODUCTION READY**

**Issues Found in Screenshot Review**:
1. ❌ Tutorial pages showing "undefined" as title
2. ❌ Instructions page showing raw markdown (wall of text)
3. ❌ Green answer blocks not visible (needed scroll to verify)

**Fixes Applied**:
1. ✅ **Added `title` field to tutorial pages** (Lines 405, 471, 522)
   - `tutorial_preanswered`: "Tutorial Example 1/4 - Correct Answers"
   - `tutorial_practice` (2 & 3): "Tutorial Example 2/4 - Practice", "Tutorial Example 3/4 - Practice"

2. ✅ **Fixed Instructions page markdown rendering** (Lines 884-886)
   - Modified `showTextPage()` to detect instructions type
   - Apply `markdownToHtml()` conversion for proper formatting
   - Result: Headers, bold text, lists now display correctly

3. ✅ **Verified green answer blocks render** (Line 1018+)
   - Green background (#e8f5e9) with green border (#4caf50)
   - All 6 answer blocks display correctly
   - Checkmarks (✓) visible on all answers

**Final File**: `test_with_tutorials.ts` (295KB, 30 pages) - PRODUCTION READY

### Session 6: Page Removal & Q2 Enhancements ✅
**USER EXPERIENCE IMPROVEMENTS**

**Changes Made**:
1. ✅ **Removed unnecessary pages** (Lines 1-3 removed)
   - Removed welcome/intro page
   - Removed consent page
   - Removed attention check page
   - Survey now starts directly with Instructions page

2. ✅ **Enhanced Q2 with detailed examples** (Lines 395-408)
   - Added comprehensive explanations for Q2.1 (perception/reasoning/action)
   - Added all 12 subcategory examples for Q2.2
   - Improved clarity with concrete examples for each category

3. ✅ **Verified Q3-Q5 examples complete**
   - Confirmed all standard questions have proper choice text and examples

**Final File**: `test_no_consent.ts` (32 pages, 2 servers)

### Session 7: Tutorial Feedback Implementation ✅
**INSTANT FEEDBACK & TIMER VERIFICATION - ALL REMAINING WORK COMPLETE**

**Implemented**:
1. ✅ **Added 9 instant feedback pages** (Lines 463-682 in generator)
   - Feedback page after each Tutorial 2 practice question
   - Q0, Q1.1, Q1.2, Q1.3, Q2.1, Q2.2, Q3, Q4, Q5
   - Pattern: Question page → Feedback page (alternating)

2. ✅ **Implemented conditional correct/incorrect logic** (Lines 1295-1379 in generated TS)
   - Green feedback (#e8f5e9 background, #4caf50 border) for correct answers
   - Red feedback (#ffebee background, #f44336 border) for incorrect answers
   - Displays human-readable text for all answer values
   - Special handling for Q0 (analysis notes) - always shows as correct with explanation

3. ✅ **Fixed O*NET ID mismatch bug** (Line 820-827)
   - Updated tutorial correct answers from old format ('15') to new format ('L1_04')
   - Fixed lookup logic to properly display human-readable O*NET classifications
   - Added lookup support for standard questions (q3, q4, q5) by referencing previous page choices

4. ✅ **Verified 20-second timer enforcement** (Lines 1524-1556 in generated TS)
   - Timer disables Next button on page load
   - Countdown displays from 20 seconds to 0
   - Next button only enabled after timer expires
   - Visual feedback: "✓ You may now continue" when ready
   - Tutorial Q0 uses 5-second timer for faster practice

**Key Features**:
- **Color-coded feedback**: Immediate visual indication of correctness
- **Educational tips**: Each feedback page includes contextual guidance
- **Human-readable text**: All answers display proper labels, not raw IDs
- **Comprehensive coverage**: Feedback for all 9 question types

**Final File**: `test_with_feedback.ts` (41 pages, 2 servers) - **FULLY PRODUCTION READY**

## Working Files

### Core Files

- **`generate_main_ts.py`** - Main generator script (fully implemented, production-ready)
- **`questions_config.csv`** - Question definitions and choices
- **`../scripting-utils/test_dynamic.html`** - Dynamic test harness (loads any .ts file via URL param)
- **`../scripting-utils/primate.js`** - Mock Gorilla API for local testing
- **`style.less`** - Gorilla styles

### Generated Files (Gitignored)

- **`test_with_tutorials.ts`** - Complete test file (295KB, 30 pages) **PRODUCTION READY**
- **`main.ts`** - Default output filename
- **`main_*.ts`** - Any custom named study files

### Data Sources

- **`../../data/initial/data_unified_filtered.json`** - MCP server data (225MB, 16,940 servers)
  - Must have: `tools` array and `canonical_description`
  - Generator filters servers with both requirements

### Documentation

- **`PROJECT_STATUS.md`** - This file (complete project documentation)
- **`../scripting-utils/iterative_survey_creation_readme.md`** - Development workflow principles
- **`../scripting-utils/instructions_gorilla_coding.md`** - Gorilla coding instructions

### Customization

#### Modify Questions

Edit `questions_config.csv`:
```csv
question_id,question_text,choice_value,choice_text
q1,Q1: Your question text?,1,Option 1 text
q1,Q1: Your question text?,2,Option 2 text
q2,Q2: Another question?,1,Yes
q2,Q2: Another question?,0,No
```

Then regenerate: `python3 generate_main_ts.py --servers 10`

#### Select Specific Servers

Modify `load_servers()` in `generate_main_ts.py`:
```python
# Add custom filters:
if server.get('stargazers_count', 0) > 10:  # Only popular
if 'finance' in description.lower():        # Only finance
if server.get('canonical_official'):        # Only official
```

## Test URLs

```bash
# Full survey with tutorials (RECOMMENDED)
http://localhost:8081/scripting-utils/test_dynamic.html?file=test_with_tutorials.ts&v=1

# Q0 only (for quick testing)
http://localhost:8081/scripting-utils/test_dynamic.html?file=test_with_q0.ts&v=1
```

## Key Commands

```bash
# Activate environment
source ~/mcp-monitoring/.venv/bin/activate

# Generate test survey with tutorials
python3 generate_main_ts.py --servers 2 --output test_with_tutorials.ts --seed 42

# Generate larger survey for production
python3 generate_main_ts.py --servers 10 --output main.ts --seed 123

# Test in browser (increment v=2, v=3 for cache-busting)
# URL: http://localhost:8081/scripting-utils/test_dynamic.html?file=test_with_tutorials.ts&v=1
```

## Production Deployment

### 1. Generate Production Survey

```bash
# Generate full study with 100 servers
python3 generate_main_ts.py --servers 100 --output production_survey.ts --seed 999
```

### 2. Quality Checks

Before deploying:
- [ ] Test with `test_dynamic.html` locally
- [ ] Verify all questions load from CSV
- [ ] Check server selection is appropriate
- [ ] Review total study length (pages)
- [ ] Test navigation (Back/Next buttons)
- [ ] Verify Q0 timer and validation works
- [ ] Check tutorial examples render correctly

### 3. Upload to Gorilla

1. Create new task in Gorilla
2. Upload `production_survey.ts` as TypeScript task
3. Upload `style.less` as task stylesheet
4. Configure task settings and metrics:
   - Add metrics: `question_id` and `response`
5. Test on Gorilla staging environment

### 4. Monitor Data Quality

After launch:
- Check attention check pass rates
- Verify Q0 timer compliance (20 seconds minimum)
- Review tutorial completion rates
- Track completion rates and drop-off points
- Analyze time-per-page distributions

### Data Collection Format

Responses stored as:
```javascript
{
  name: "abusech-mcp_q1",      // {serverName}_{questionId}
  value: "1",                   // Selected choice value
  checked: "1",
  servername: "abusech-mcp",
  question: "q1"
}
```

## Implementation Details

### Q0 Analysis Notes
- **Location**: Lines 662-676 (data), 974-1006 (display), 1279-1296 (validation)
- **Features**: Textarea (min 10 chars), 20-second timer, disabled Next button
- **Position**: First question for each server (Question 0/9)

### Tutorial Examples
- **Location**: Lines 381-551 (data), 854-859 (handlers), 1017-1071 (display)
- **Structure**:
  - Example 1: Pre-answered (asher-mcp) with green answer blocks
  - Example 2: Practice (base-mcp) with blue banner
  - Example 3: Practice (DesktopCommanderMCP) with blue banner
- **Pages**: 7 tutorial pages total (3 intros + 1 preanswered + 3 practice)

### Instruction Page
- **Location**: Lines 331-378
- **Content**: O*NET classification (3 levels) and Functionality classification (2 levels)
- **Position**: After consent, before tutorials

## Key Lessons Learned

1. **Never manually edit .ts files** - Always regenerate from `generate_main_ts.py`
2. **Unicode breaks TypeScript** - Use ASCII characters only
3. **Cache-busting required** - Use `?v=1`, `?v=2` etc. to force browser reload
4. **Browser testing essential** - Visual verification catches issues code review misses
5. **Incremental development** - Test with 2 servers first, then scale up

## Survey Flow

```
1. Intro page
2. Attention check (select "No AI use at all")
3. Consent (4 checkboxes)
4. Instructions page
5-11. Tutorial examples (7 pages)
   - Tutorial 1 intro → Pre-answered example
   - Tutorial 2 intro → Practice example
   - Tutorial 3 intro → Practice example
12+. Real survey (per server):
   - Q0: Analysis Notes (20-second timer)
   - Q1.1: O*NET Level 1
   - Q1.2: O*NET Level 2 (conditional)
   - Q1.3: O*NET Task (conditional)
   - Q2.1: Functionality Main
   - Q2.2: Functionality Sub (conditional)
   - Q3-Q5: Standard questions
Final. Completion page
```

## Success Criteria ✅

- ✅ Generator produces valid TypeScript
- ✅ Instruction page displays correctly
- ✅ Q0 analysis notes with 20-second timer
- ✅ Tutorial examples render correctly (browser-verified)
- ✅ Full survey navigates from intro to completion
- ✅ All features validated with screenshots

## Potential UI/UX Improvements

Based on the implementation, consider these enhancements for future iterations:

### Tutorial Enhancements
1. **Visual Hierarchy**: Add progress indicator showing "Tutorial 1 of 3" to help users understand tutorial length
2. **Answer Highlighting**: In pre-answered tutorial, use larger font or bolder styling for the correct answers to improve readability
3. **Practice Feedback**: Add a "Check Answers" button after practice tutorials to give immediate feedback (optional feature)
4. **Tutorial Skip Option**: Add "Skip Tutorials" button for repeat participants (with warning)

### Q0 Analysis Notes
1. **Timer Visibility**: Consider making timer larger or more prominent (current: inline text)
2. **Character Counter**: Add live character count to help users meet the 10-character minimum (e.g., "10/10 characters")
3. **Timer Sound**: Optional audio notification when timer completes (subtle ping)
4. **Example Text**: Add a collapsed "See example" section showing what good analysis notes look like

### Instructions Page
1. **Interactive Examples**: Consider adding clickable tooltips on classification terms for more detail
2. **Print/Save Option**: Add button to download instructions as PDF for reference during survey
3. **Search Function**: For longer classification lists, add search/filter capability

### General Improvements
1. **Progress Bar**: Add overall survey progress indicator at top (currently only page X/Y shown)
2. **Save & Resume**: Implement session storage to allow participants to resume if they close browser
3. **Mobile Responsiveness**: Test and optimize for tablet/mobile devices
4. **Accessibility**: Add ARIA labels, keyboard navigation, screen reader support
5. **Color Contrast**: Verify WCAG AA compliance for all colored elements (green answer blocks, blue practice banner)

### Data Quality Enhancements
1. **Attention Checks**: Add 1-2 more attention checks throughout survey (not just at start)
2. **Time Tracking**: Log time spent on each page to identify rushing or confusion
3. **Response Validation**: Add soft warnings for suspiciously short analysis notes (e.g., "Are you sure? Most participants write more.")

### Next Steps for Testing
1. **User Testing**: Conduct pilot with 5-10 participants to gather feedback
2. **A/B Testing**: Test different tutorial formats to see which improves classification accuracy
3. **Analytics**: Track completion rates, drop-off points, and time-per-page distributions

## Troubleshooting

### "Error: test_with_tutorials.ts not found"

- Ensure the `.ts` file exists in `scripting-main` directory (or adjust URL path)
- Check `file` parameter in URL matches filename exactly
- Verify local server is running: `python3 -m http.server 8081`

### "Loaded 0 servers"

- Check `data_unified_filtered.json` path is correct
- Verify servers have both `tools` and `canonical_description`
- Try with `--servers 1` to debug

### Questions not loading

- Check `questions_config.csv` format
- Ensure CSV has header: `question_id,question_text,choice_value,choice_text`
- Verify no special characters breaking CSV parsing

### Navigation not working

- Check browser console for JavaScript errors
- Verify jQuery loaded (should be in Gorilla)
- Check button IDs match event handlers

### Cache issues in browser

- Use `?v=1`, `?v=2`, `?v=3` etc. in URL to force reload
- Clear browser cache completely
- Try incognito/private browsing mode

---

## Key Lessons Learned

1. **Never manually edit .ts files** - Always regenerate from `generate_main_ts.py`
2. **Unicode breaks TypeScript** - Use ASCII characters only in generated content
3. **Cache-busting required** - Use `?v=1`, `?v=2` etc. to force browser reload
4. **Browser testing essential** - Visual verification catches issues code review misses
5. **Incremental development** - Test with 2 servers first, then scale to production
6. **Generator-based workflow** - Much easier than manual TypeScript editing
7. **Dynamic test harness** - `test_dynamic.html` allows testing any .ts file without rebuilding

---

*Complete project documentation - Generator-based workflow for Gorilla MCP classification studies*

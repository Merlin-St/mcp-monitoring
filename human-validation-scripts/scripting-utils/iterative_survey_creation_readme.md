# Iterative Survey Creation Process

## Development Workflow for AI Agents

### Core Principle: Visual Verification First
**ALWAYS use Playwright to visually test surveys before considering them complete.** Never assume code works without seeing it render.

### The Iteration Loop

1. **Generate** → Create/modify TypeScript survey file using `generate_main_ts.py`
2. **View** → Open in Playwright: `http://localhost:8081/scripting-utils/test_dynamic.html?file=YOUR_FILE.ts`
3. **Navigate** → Click through the entire survey to verify all pages
4. **Verify** → Check server info, questions, choices, navigation. Take full-page-screenshots of most important new pages and then save them to .playwright-mcp and inspect them.
5. **Fix** → If issues found, update generator script or CSV
6. **Repeat** → Go back to step 1 until perfect

### Key Commands

```bash
# Generate survey
python3 generate_main_ts.py --servers 2 --output main_test.ts

# Test with Playwright (use this EVERY TIME)
# In Claude Code: use mcp__playwright__browser_navigate
http://localhost:8081/scripting-utils/test_dynamic.html?file=main_test.ts
```

### Critical Files

- `../scripting-main/generate_main_ts.py` - Generator script (modify logic here)
- `../scripting-main/questions_config.csv` - Questions/choices (edit to change survey)
- `test_dynamic.html` - Dynamic test page (loads any .ts file via `?file=` param)
- `../../data/initial/data_unified_filtered.json` - Server data source

### Why Dynamic Testing?

The `test_dynamic.html` file accepts `?file=` URL parameter, so you can test ANY generated .ts file without rebuilding HTML:
- `?file=main.ts` - Default
- `?file=main_test.ts` - Test version
- `?file=main_pilot.ts` - Pilot study
- `?file=main_batch1.ts` - Batch 1

This enables rapid iteration: generate → test → fix → repeat.

### Gotchas for Future Agents

1. **Never skip Playwright visual testing** - Code that compiles doesn't mean UI works
2. **Questions come from CSV** - Don't hardcode them in Python
3. **Use URL parameters** - No need to rebuild HTML for each .ts file
4. **Test full navigation** - Click through intro → all servers → completion
5. **Check truncation** - Verify long descriptions/tool names display properly

## Critical Debugging Lessons

### NEVER Manually Edit Generated .ts Files
- **Always modify `generate_main_ts.py`**, never edit `main.ts` directly
- Manual edits to .ts files will be lost on next generation
- If you need to test something, modify the generator and regenerate

### Browser Caching Issues
- Browser aggressively caches TypeScript files
- **Solution**: Use cache-busting URL params (e.g., `?file=main_test.ts&v=2`)
- Close and reopen browser between major changes
- If seeing stale errors, increment version number in URL

### Unicode Characters Break TypeScript Transpilation
- **Never use Unicode arrows (→)** or special characters in template strings
- Use ASCII equivalents: Replace `→` with `:` or `->`
- TypeScript transpiler in browser will fail silently with "Unexpected token" errors
- Check for non-ASCII characters if you get compilation errors

### Testing New Features
1. **Start with minimal test** - Generate 2-server survey to test faster
2. **Use seed for reproducibility** - `--seed 42` ensures same servers each time
3. **Test incrementally** - Add one feature at a time, test, then add next
4. **Compare with working version** - Keep a known-good .ts file for comparison

### Common Error Patterns
- `SyntaxError: Unexpected token ':'` → Check for Unicode characters in template strings
- `Failed to load study file` + 404 → Check file exists and server is running (port 8081)
- Survey loads but page missing → Forgot to add handler in `showPage()` or `validatePage()`
- Wrong page count → Forgot to update `total_questions_per_server` calculation

## Critical Workflow Principles

### The Golden Rule: Generator-Only Modifications
**NEVER manually edit .ts files directly. ALWAYS modify `generate_main_ts.py` and regenerate.**

Why this matters:
- Manual edits are lost on next generation
- Creates confusion between working and generated versions
- Makes debugging impossible (which version is which?)
- Wastes hours fixing the wrong file

**Correct workflow:**
1. Modify `generate_main_ts.py` to add feature
2. Generate new .ts file: `python3 generate_main_ts.py --servers 2 --output test.ts --seed 42`
3. Test in browser: `http://localhost:8081/scripting-utils/test_dynamic.html?file=test.ts&v=1`
4. If broken, debug and fix `generate_main_ts.py` (not the .ts file!)
5. Regenerate and test again

**Wrong workflow (DO NOT DO THIS):**
1. Manually edit main.ts to add feature ❌
2. It works! ❌
3. Later regenerate from generator ❌
4. All manual edits lost, survey broken ❌

### When Generated Files Fail But Manual Edits Work
This indicates a fundamental bug in `generate_main_ts.py`, not the TypeScript:
- The generator is creating malformed TypeScript code
- Manual edits bypass the generator's bug
- **Solution**: Fix the generator, don't work around it with manual edits

Common generator bugs:
- Incorrect string escaping (e.g., backticks, quotes)
- Unicode characters in template strings
- Missing or extra commas in arrays
- Incorrect template string interpolation

### Testing Strategy
1. **Always start small**: Test with 2 servers (`--servers 2`) for faster iteration
2. **Use seed for reproducibility**: Same seed = same servers = consistent testing
3. **Incremental development**: Add ONE feature at a time, test, then add next
4. **Keep a reference**: Maintain a known-good .ts file for comparison
5. **Version your tests**: Use `?file=test_v1.ts`, `test_v2.ts`, etc.

### Debug Efficiently
**If TypeScript file won't load:**
1. Check for Unicode characters (especially →, •, etc.)
2. Verify server is running on port 8081
3. Try cache-busting: `?file=test.ts&v=2`
4. Compare with working file (byte-by-byte if needed)
5. Test with minimal 2-server version
6. Check browser console for actual error message

**If specific page won't render:**
1. Check handler exists in `showPage()` function
2. Check validation exists in `validatePage()` function
3. Verify page type string matches exactly (case-sensitive)
4. Use Playwright to navigate and screenshot the error

### File Organization
- `main.ts` - Production survey (generated, never manually edit)
- `main_test.ts` - Test version with fewer servers (generated)
- `main_v*.ts` - Version snapshots (generated, for comparison)
- `generate_main_ts.py` - **THE ONLY FILE TO EDIT**
- `questions_config.csv` - Question definitions (edit as needed)

### Context Management
When context runs low:
1. Document current state in markdown files
2. Take screenshots of working features
3. Commit generator changes to git
4. Write clear todo list for continuation
5. **Never leave manual edits as the only working version**

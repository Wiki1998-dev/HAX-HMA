# research_scout.py — drop this in your project root
import os, subprocess, yake

# Step 1: Read your entire codebase
code_text = ""
for root, _, files in os.walk("."):
    for f in files:
        if f.endswith((".py", ".js", ".ts", ".md", ".txt")):
            try:
                code_text += open(os.path.join(root, f)).read() + "\n"
            except: pass

# Step 2: Extract keywords with YAKE (unsupervised, no API)
kw_extractor = yake.KeywordExtractor(
    lan="en", n=3,        # up to 3-word phrases
    dedupLim=0.7,
    top=10                # top 10 topics
)
keywords = [kw for kw, score in kw_extractor.extract_keywords(code_text)]
print("Detected topics:", keywords)

# Step 3: Search papers for each keyword
for kw in keywords:
    print(f"\n🔍 Searching: {kw}")
    subprocess.run([
        "lixplore", "-A", "-q", kw,
        "-m", "5", "--sort", "newest",
        "-X", "csv", "-o", f"papers_{kw.replace(' ','_')}.csv"
    ])
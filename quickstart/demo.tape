# vhs tape for the xk6-llm demo. Renders to quickstart/img/demo.gif.

Output quickstart/img/demo.gif

Set FontSize 14
Set Width 1280
Set Height 720
Set Theme "Catppuccin Mocha"
Set Framerate 12
Set PlaybackSpeed 1.6
Set TypingSpeed 25ms

Type "# xk6-llm: load test any OpenAI-compatible inference server"
Sleep 500ms
Enter
Type "# every panel - TTFT, ITL, TPOT, goodput, cost, energy - emitted live"
Sleep 500ms
Enter
Type "clear"
Enter

Type "./build/k6 version"
Enter
Sleep 2000ms

Type "# 3-turn conversations, prefix-cache tagged, real cost + energy"
Sleep 500ms
Enter
Type "./build/k6 run --log-format=raw --quiet --vus 4 --iterations 12 -e LLM_BASE_URL=http://localhost:11434/v1 -e LLM_MODEL=qwen2.5:0.5b examples/showcase.js"
Enter

Sleep 60s

Type "# dashboard: http://localhost:3000/d/xk6-llm"
Sleep 3s
Enter

# vhs tape for the xk6-llm demo. Renders to quickstart/img/demo.gif.

Output quickstart/img/demo.gif

Set FontSize 14
Set Width 1280
Set Height 760
Set Theme "Catppuccin Mocha"
Set PlaybackSpeed 1.6
Set TypingSpeed 25ms

Type "# xk6-llm: load test any OpenAI-compatible inference server"
Sleep 600ms
Enter
Type "# emits TTFT, ITL, TPOT, goodput, cost, energy. ships to Grafana."
Sleep 600ms
Enter
Type "clear"
Enter

Type "./build/k6 version"
Enter
Sleep 2500ms

Type "# fire showcase: 3-turn sessions, ramping arrival, cost + energy + SLOs"
Sleep 800ms
Enter
Type 'export K6_PROMETHEUS_RW_TREND_STATS="p(50),p(95),p(99),min,max,avg,count,sum"'
Enter
Sleep 600ms
Type "./build/k6 run -o experimental-prometheus-rw=http://localhost:9090/api/v1/write -e LLM_BASE_URL=http://localhost:11434/v1 -e LLM_MODEL=qwen2.5:0.5b examples/showcase.js"
Enter

Sleep 140s

Type "# dashboard live at http://localhost:3000/d/xk6-llm"
Sleep 4s
Enter

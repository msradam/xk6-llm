## vhs tape: records a polished GIF of the xk6-llm showcase run.
## Build:  vhs quickstart/demo.tape  ->  quickstart/img/demo.gif
##
## Assumes a vLLM endpoint is already reachable. Set the env vars on the
## host running vhs:  LLM_BASE_URL, LLM_MODEL.

Output quickstart/img/demo.gif

Set FontSize 13
Set Width 1280
Set Height 760
Set Theme "Catppuccin Mocha"
Set PlaybackSpeed 1.2
Set TypingSpeed 35ms

Hide
Type "cd quickstart && docker compose up -d >/dev/null 2>&1 && cd .."
Enter
Sleep 2s
Show

Type "# xk6-llm: load test any OpenAI-compatible inference server"
Sleep 800ms
Enter
Type "# emits TTFT, ITL, TPOT, goodput, cost, energy"
Sleep 800ms
Enter
Type "clear"
Enter

Type "./build/k6 version | head -3"
Enter
Sleep 2s

Type "# fire the showcase: ramping arrival, 3-turn sessions, cost + energy + SLOs"
Sleep 1s
Enter
Type "K6_PROMETHEUS_RW_TREND_STATS='p(50),p(95),p(99),min,max,avg,count,sum' ./build/k6 run \"
Enter
Type "  -o experimental-prometheus-rw=http://localhost:9090/api/v1/write \"
Enter
Type "  -e LLM_BASE_URL=$LLM_BASE_URL -e LLM_MODEL=$LLM_MODEL \"
Enter
Type "  examples/showcase.js"
Enter

Sleep 130s

Type "# open the dashboard:  http://localhost:3000/d/xk6-llm"
Sleep 4s
Enter

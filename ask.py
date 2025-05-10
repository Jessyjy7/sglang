#!/usr/bin/env python3
from sglang import set_default_backend, function, system, user, assistant, gen
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
import requests

# 1) point at your server
set_default_backend(RuntimeEndpoint("http://localhost:30000"))

# 2) wrap the model into a chat function
@function
def step(s, msg):
    # you can inject a system prompt only once at the very top:
    if not s:
        s += system("You are a helpful assistant.")
    # add the user’s message
    s += user(msg)
    # get the model’s reply
    s += assistant(gen("reply", max_tokens=128))
    return s

def call_weather_api(location: str) -> str:
    # example external API call; replace with any service you like
    r = requests.get(f"https://api.example.com/weather?loc={location}")
    data = r.json()
    return data["forecast"]

def main():
    transcript = ""    # this is the growing “s” we pass in
    state = None

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit","quit"}:
            break

        # example of calling a real API mid-stream:
        if q.startswith("weather in "):
            loc = q[len("weather in "):]
            forecast = call_weather_api(loc)
            print(f"(weather API says: {forecast})")
            # spin that into the transcript as if the assistant told you
            transcript += f"SYSTEM: Weather for {loc} is {forecast}\n"

        # run one model step
        if state is None:
            state = step(q)         # first turn; s="" under the hood
        else:
            # feed the entire transcript back in
            state = step(transcript + q)

        # grab out the model’s reply
        bot = state["reply"]
        print("Bot:", bot)

        # append to transcript for next turn
        transcript += f"USER: {q}\nASSISTANT: {bot}\n"

if __name__ == "__main__":
    main()
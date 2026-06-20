#!/usr/bin/env python3
# AmongUs.py - A totally serious Among Us tool

import webbrowser
import sys
import time

def main():
    # The "secret" video link
    video_url = "https://youtu.be/rZNDsiW7wpk?si=hHPFcRDwB3h5kJZW"
    
    # Funny loading messages
    messages = [
        "🔍 Scanning for impostors...",
        "🎮 Loading Among Us assets...",
        "🕵️‍♂️ Checking your electrical tasks...",
        "💀 Detecting sus activity...",
        "📡 Connecting to The Skeld...",
        "🔪 Vent cleaning in progress...",
        "👾 Sabotaging your browser...",
    ]
    
    print("\n🚀 AMONG US LAUNCHER v1.0 🚀")
    print("=" * 35)
    
    for msg in messages:
        print(f"{msg}", end=" ", flush=True)
        time.sleep(0.4)
        print("✅")
        time.sleep(0.2)
    
    print("\n⚠️  CRITICAL WARNING ⚠️")
    print("An impostor has been detected in your system!")
    print("Opening emergency meeting in your browser...\n")
    
    time.sleep(1)
    
    # The big reveal - open the video
    print("🎬 Playing: 'Among Us - Short Animation - GOD DAMN IT'")
    webbrowser.open(video_url)
    
    print("\n💀 You have been pranked!")
    print("👋 Thanks for playing. Now get back to work!")
    print("(Press ENTER to exit this impostor...)", end="")
    input()
    sys.exit(0)

if __name__ == "__main__":
    main()
"""Plan + code inference for a single raw video (no simulator).

Usage (in the `mediapipe` conda env):
    # simplest: name the objects in the scene
    python run_video.py test-input-video/pour-tea.mp4 \
        --objects "brown teapot,left white cup,right white cup,wooden tray"

    # or reuse a task JSON (e.g. a benchmark file, or one written by a previous run)
    python run_video.py test-input-video/pick-cup-1.mp4 \
        --data_json test-output/pick-cup-1/pick-cup-1.json

Outputs go to test-output/<video stem>/ unless --out_dir is given:
    <stem>.json           task JSON actually used (written when --objects is given)
    <stem>/               frames, hand overlays, keyframelist.json, frames_with_hand.json
    <model>-<model>/      Result.json (plan + code), Planlogs.json, Codelogs.json, ...

The task JSON only needs "object_list": the closed set of object names the
agents may use (the plan and the code must use these exact strings). The
optional "positions" dict is only read by the simulator, which is stubbed here.

Differences from run.py:
  * API key comes from .env (OPENAI_API_KEY) via load_dotenv(); endpoint is OpenAI.
  * The non-standard "safe_mode" payload field is stripped (OpenAI rejects it).
  * The simulator Execute tool is replaced by a no-op stub (no pybullet/moviepy).
  * Frame extraction uses get_frames.FrameExtractor, but hand landmarks are drawn
    with cv2 because mediapipe>=0.10.30 no longer ships `mediapipe.solutions`.
"""
import argparse
import json
import os
import sys
import time
import types
import warnings
from collections import defaultdict
from datetime import datetime

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(REPO))  # so `import LongVIL...` works
sys.path.insert(0, REPO)
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv(os.path.join(REPO, ".env"))


# ---------------------------------------------------------------- frames
def extract_frames(video, out_dir):
    import cv2
    import numpy as np

    # get_frames.py imports legacy mediapipe modules at import time; provide
    # placeholders (they are only used in process_frame_results, overridden below).
    fake_solutions = types.ModuleType("mediapipe.solutions")
    sys.modules.setdefault("mediapipe.solutions", fake_solutions)
    for name in ["mediapipe.framework", "mediapipe.framework.formats"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules.setdefault("mediapipe.framework.formats.landmark_pb2", types.ModuleType("landmark_pb2"))
    import mediapipe as mp
    mp.solutions = fake_solutions
    import get_frames

    HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                        (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

    class Extractor(get_frames.FrameExtractor):
        def process_frame_results(self, rgb_image, detection_result):
            img = np.copy(rgb_image)
            h, w, _ = img.shape
            ax, ay, hands = {}, {}, set()
            for lms, hd in zip(detection_result.hand_landmarks, detection_result.handedness):
                pts = [(int(l.x * w), int(l.y * h)) for l in lms]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(img, pts[a], pts[b], (255, 255, 255), 2)
                for p in pts:
                    cv2.circle(img, p, 3, (0, 0, 255), -1)
                label = hd[0].category_name
                xs, ys = [l.x for l in lms], [l.y for l in lms]
                ax[label] = np.average(xs) * w
                ay[label] = np.average(ys) * h
                hands.add(label)
                cv2.putText(img, label, (int(min(xs) * w), int(min(ys) * h) - self.MARGIN),
                            cv2.FONT_HERSHEY_DUPLEX, self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR,
                            self.FONT_THICKNESS, cv2.LINE_AA)
            return img, ax, ay, hands

    cwd = os.getcwd()
    os.chdir(REPO)  # hand_landmarker.task is loaded via a relative path
    try:
        Extractor(os.path.abspath(os.path.join(cwd, video)) if not os.path.isabs(video) else video,
                  out_dir, gaussian_sigma=5, prominence=0.8).extract_frames()
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------- agents
def patch_for_openai_and_no_sim():
    import LongVIL.tools.api_utils as api_utils

    orig = api_utils.request_chat_completion

    def request_chat_completion(base_url, headers, payload):
        payload = {k: v for k, v in payload.items() if k != "safe_mode"}
        if not payload.get("stop"):
            payload.pop("stop", None)
        for attempt in range(4):
            try:
                return orig(base_url, headers, payload)
            except Exception as e:  # transient network / rate limit
                body = getattr(getattr(e, "response", None), "text", "")
                print(f"[api] attempt {attempt + 1} failed: {e} {body[:500]}")
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))

    api_utils.request_chat_completion = request_chat_completion

    from transformers import Tool

    class Execute(Tool):
        name = "Execute"
        description = "A tool that executes parsed Python-like robot control code in a simulated environment."
        inputs = {"codelist": {"description": "List of Python-style robot control code blocks", "type": "string"}}
        output_type = "string"

        def __init__(self, args):
            super().__init__()

        def forward(self, codelist):
            return "Execute Successfully!"  # simulation intentionally skipped

    stub = types.ModuleType("LongVIL.tools.execute")
    stub.Execute = Execute
    sys.modules["LongVIL.tools.execute"] = stub


def jsonl_to_dict(path):
    d = defaultdict(list)
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                o = json.loads(line)
                d[o["key"]].append(o["tokens"])
    return dict(d)


def load_task(a, stem):
    """Return the task dict from --data_json, or build one from --objects."""
    if a.data_json:
        data = json.load(open(a.data_json, encoding="utf-8"))
        if a.objects:
            print("note: --objects overrides object_list from --data_json")
    else:
        data = {"id": stem, "positions": {}}
    if a.objects:
        data["object_list"] = [o.strip() for o in a.objects.split(",") if o.strip()]
        path = os.path.join(a.out_dir, f"{stem}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print("task JSON written:", path)
    if not data.get("object_list"):
        raise SystemExit("No objects given: pass --objects \"a,b,c\" or a --data_json with object_list.")
    data.setdefault("positions", {})
    return data


def run_agents(a, frame_dir, data):
    patch_for_openai_and_no_sim()
    from LongVIL.agent.Planagent import create_Plan_agent
    from LongVIL.agent.Codeagent import create_Code_agent

    args = argparse.Namespace(
        engine_type="gpt",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1/chat/completions",
        model_name=a.model,
        llm_model_name=a.model,
        max_iterations=30,
        data_type="clean",
        use_reflection="y",
        video_path=os.path.join(frame_dir, "frames"),
        object_list=data["object_list"],
        object_dict=data["positions"],
        key_frames=json.load(open(os.path.join(frame_dir, "keyframelist.json"))),
        frames_with_hand=json.load(open(os.path.join(frame_dir, "frames_with_hand.json"))),
    )
    args.save_path = os.path.join(a.out_dir, f"{a.model}-{a.model}")
    os.makedirs(args.save_path, exist_ok=True)
    os.environ["TOKEN_PATH"] = os.path.join(args.save_path, "token_usage.jsonl")
    print("keyframes:", args.key_frames, "\nobjects:", args.object_list, "\nsave_path:", args.save_path)

    start_wall, t0 = datetime.now(), time.perf_counter()
    plan_agent, code_agent = create_Plan_agent(args), create_Code_agent(args)

    plan_agent.run(
        "You will be given a list of keyframes of a video (parameter key_frames). "
        "Parameter object_list represents the name of the object appearing in the video. "
        "The parameter save_path indicates the path where the results are saved. "
        "Please analyzes the sequence of video frames to recognize and describe human-object interaction actions as required.",
        key_frames=args.key_frames, object_list=args.object_list, save_path=args.save_path,
    )
    json.dump(plan_agent.write_inner_memory_from_logs(),
              open(f"{args.save_path}/Planlogs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    action_list = json.load(open(f"{args.save_path}/Result.json"))["action_sequences"]
    code_agent.run(
        "You will be given a action list. Parameter action_list represents the sequence of actions you need to process. "
        "The parameter save_path indicates the path where the results are saved. "
        "Please perform code conversion, execution and result verification as required.",
        action_list=action_list, save_path=args.save_path,
    )
    json.dump(code_agent.write_inner_memory_from_logs(),
              open(f"{args.save_path}/Codelogs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    json.dump({"start_time": start_wall.isoformat(timespec="seconds"),
               "end_time": datetime.now().isoformat(timespec="seconds"),
               "duration_seconds": time.perf_counter() - t0},
              open(os.path.join(args.save_path, "runtime.json"), "w"), indent=2)
    json.dump(jsonl_to_dict(os.environ["TOKEN_PATH"]),
              open(os.path.join(args.save_path, "token_usage_dict.json"), "w"), indent=2)
    print("\n==== Result.json ====")
    print(open(f"{args.save_path}/Result.json").read())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LongVIL plan + code inference on one video (no simulator).")
    p.add_argument("video", help="path to the demonstration video (.mp4)")
    p.add_argument("--objects", help='comma-separated object names, e.g. "white bowl,red cube"')
    p.add_argument("--data_json", help="task JSON with an object_list (alternative to --objects)")
    p.add_argument("--out_dir", help="default: test-output/<video stem>")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--skip_frames", action="store_true", help="reuse frames from a previous run")
    a = p.parse_args()
    if not (a.objects or a.data_json):
        p.error("give --objects or --data_json")

    stem = os.path.splitext(os.path.basename(a.video))[0]
    a.out_dir = os.path.abspath(a.out_dir or os.path.join(REPO, "test-output", stem))
    os.makedirs(a.out_dir, exist_ok=True)
    data = load_task(a, stem)
    frame_dir = os.path.join(a.out_dir, stem)
    if not a.skip_frames:
        extract_frames(a.video, a.out_dir)
    run_agents(a, frame_dir, data)

import requests
from nltk.translate.bleu_score import sentence_bleu
import time
import threading
import psutil
import subprocess
import pyttsx3
import json

# Sample test data: queries and ground truth answers related to Disneyland
test_data = [
    {"query": "What are the FastPass rules?", "ground_truth": "FastPass allows you to reserve a time slot for attractions."},
    {"query": "Where is Space Mountain located?", "ground_truth": "Space Mountain is in Tomorrowland."},
    {"query": "What time does Disneyland open?", "ground_truth": "Disneyland typically opens at 8:00 AM."},
    # Add more queries as needed (e.g., up to 10 or more for comprehensive testing)
]

# Server URL (adjust as needed if the server runs on a different address/port)
server_url = "http://localhost:8000/query"

def send_query(query):
    """Send a query to the server and return the response."""
    try:
        response = requests.post(server_url, json={"text": query}, timeout=10)
        response.raise_for_status()
        return response.json()["answer"]
    except requests.RequestException as e:
        print(f"Error sending query '{query}': {e}")
        return None

def calculate_bleu(response, ground_truth):
    """Calculate BLEU score for the response against the ground truth."""
    if response is None:
        return 0
    reference = [ground_truth.split()]
    candidate = response.split()
    return sentence_bleu(reference, candidate)

def monitor_resources(interval=0.5):
    """Monitor CPU, memory, and GPU usage in a background thread."""
    data = {"cpu": [], "memory": [], "gpu_vram": []}
    stop_event = threading.Event()

    def collect():
        while not stop_event.is_set():
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB
            try:
                gpu_output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
                )
                gpu_vram = int(gpu_output.decode().strip()) / 1024  # Convert MB to GB
            except Exception:
                gpu_vram = None  # Handle cases where nvidia-smi is unavailable
            data["cpu"].append(cpu)
            data["memory"].append(memory)
            data["gpu_vram"].append(gpu_vram)
            time.sleep(interval)

    thread = threading.Thread(target=collect)
    thread.start()
    return data, stop_event

def assess_accuracy(query, response, num_assessors=3):
    """Collect accuracy ratings from human assessors."""
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    ratings = []
    for i in range(num_assessors):
        while True:
            try:
                rating = int(input(f"Assessor {i+1}, rate the response accuracy (1-5): "))
                if 1 <= rating <= 5:
                    ratings.append(rating)
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    return ratings

def generate_tts(text):
    """Generate and play TTS audio for the given text."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)  # Speech rate
        engine.setProperty("volume", 0.9)  # Volume (0.0 to 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error generating TTS: {e}")

def assess_tts_quality(text, num_participants=5):
    """Collect TTS quality ratings from participants."""
    generate_tts(text)
    ratings = []
    for i in range(num_participants):
        while True:
            try:
                rating = int(input(f"Participant {i+1}, rate the TTS quality (1-5): "))
                if 1 <= rating <= 5:
                    ratings.append(rating)
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    return sum(ratings) / len(ratings) if ratings else None

def main():
    """Main function to orchestrate the evaluation process."""
    results = []

    # Start resource monitoring
    print("Starting resource monitoring...")
    resource_data, stop_event = monitor_resources()

    # Process queries: send to server, calculate BLEU, measure response time
    print("\nProcessing queries...")
    for item in test_data:
        query = item["query"]
        ground_truth = item["ground_truth"]
        start_time = time.time()
        response = send_query(query)
        response_time = time.time() - start_time
        if response is None:
            continue
        bleu = calculate_bleu(response, ground_truth)
        results.append({
            "query": query,
            "ground_truth": ground_truth,
            "response": response,
            "bleu_score": bleu,
            "response_time": response_time,
            "accuracy_ratings": [],
            "tts_rating": None
        })

    # Stop resource monitoring
    stop_event.set()
    time.sleep(1)  # Allow time for the last data collection
    print("Resource monitoring stopped.")

    # Calculate resource usage metrics
    cpu_avg = sum(resource_data["cpu"]) / len(resource_data["cpu"]) if resource_data["cpu"] else 0
    memory_max = max(resource_data["memory"]) if resource_data["memory"] else 0
    gpu_vram_max = max(resource_data["gpu_vram"]) if resource_data["gpu_vram"] else 0

    # Accuracy assessment: collect human ratings
    print("\nStarting accuracy assessment...")
    for result in results:
        ratings = assess_accuracy(result["query"], result["response"])
        result["accuracy_ratings"] = ratings

    # TTS quality assessment: select a subset (e.g., first 10) and collect ratings
    tts_subset = results[:10]  # Adjust subset size as needed
    print("\nStarting TTS quality assessment...")
    for result in tts_subset:
        rating = assess_tts_quality(result["response"])
        result["tts_rating"] = rating

    # Calculate overall metrics
    bleu_scores = [r["bleu_score"] for r in results]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    accuracy_ratings = [sum(r["accuracy_ratings"]) / len(r["accuracy_ratings"]) 
                       for r in results if r["accuracy_ratings"]]
    avg_accuracy = sum(accuracy_ratings) / len(accuracy_ratings) if accuracy_ratings else 0

    response_times = [r["response_time"] for r in results]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0

    tts_ratings = [r["tts_rating"] for r in tts_subset if r["tts_rating"] is not None]
    avg_tts_rating = sum(tts_ratings) / len(tts_ratings) if tts_ratings else 0

    # Output results
    print(f"\n--- Evaluation Summary ---")
    print(f"Average BLEU score: {avg_bleu:.2f}")
    print(f"Average accuracy rating: {avg_accuracy:.2f}")
    print(f"Average response time: {avg_response_time:.2f} seconds")
    print(f"Average CPU usage: {cpu_avg:.2f}%")
    print(f"Max memory usage: {memory_max:.2f} GB")
    print(f"Max GPU VRAM usage: {gpu_vram_max:.2f} GB" if gpu_vram_max else "Max GPU VRAM usage: N/A")
    print(f"Average TTS quality rating: {avg_tts_rating:.2f}")

    # Save detailed results to a JSON file
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "results": results,
            "summary": {
                "avg_bleu": avg_bleu,
                "avg_accuracy": avg_accuracy,
                "avg_response_time": avg_response_time,
                "avg_cpu": cpu_avg,
                "max_memory": memory_max,
                "max_gpu_vram": gpu_vram_max,
                "avg_tts_rating": avg_tts_rating
            }
        }, f, indent=4)
    print("\nDetailed results saved to 'evaluation_results.json'.")

if __name__ == "__main__":
    main()
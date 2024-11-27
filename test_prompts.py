from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_PATH = "/fsx/loubna/projects/alignment-handbook/recipes/cosmo2/sft/data"
TEMPERATURE = 0.2
TOP_P = 0.9

CHECKPOINT = "loubnabnl/smollm-350M-instruct-add-basics"

print(f"💾 Loading the model and tokenizer: {CHECKPOINT}...")
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model_s = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(device)

print("🧪 Testing single-turn conversations...")
L = [
    "Hi",
    "Hello",
    "Tell me a joke",
    "Who are you?",
    "What's your name?",
    "How do I make pancakes?",
    "Can you tell me what is gravity?",
    "What is the capital of Morocco?",
    "What's 2+2?",
    "Hi, what is 2+1?",
    "What's 3+5?",
    "Write a poem about Helium",
    "Hi, what are some popular dishes from Japan?",
]


for i in range(len(L)):
    print(f"🔮 {L[i]}")
    messages = [{"role": "user", "content": L[i]}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model_s.generate(
        inputs, max_new_tokens=200, top_p=TOP_P, do_sample=True, temperature=TEMPERATURE
    )
    with open(
        f"{BASE_PATH}/{CHECKPOINT.split('/')[-1]}_temp_{TEMPERATURE}_topp{TOP_P}.txt",
        "a",
    ) as f:
        f.write("=" * 50 + "\n")
        f.write(tokenizer.decode(outputs[0]))
        f.write("\n")


print("🧪 Now testing multi-turn conversations...")
# Multi-turn conversations
messages_1 = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
    {"role": "user", "content": "What's 2+2?"},
]
messages_2 = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "Why?"},
]
messages_3 = [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am an AI assistant. How can I help you today?"},
    {"role": "user", "content": "What's your name?"},
]
messages_4 = [
    {"role": "user", "content": "Tell me a joke"},
    {"role": "assistant", "content": "Sure! Why did the tomato turn red?"},
    {"role": "user", "content": "Why?"},
]
messages_5 = [
    {"role": "user", "content": "Can you tell me what is gravity?"},
    {
        "role": "assistant",
        "content": "Sure! Gravity is a force that attracts objects toward each other. It is what keeps us on the ground and what makes things fall.",
    },
    {"role": "user", "content": "Who discovered it?"},
]
messages_6 = [
    {"role": "user", "content": "How do I make pancakes?"},
    {
        "role": "assistant",
        "content": "Sure! Here is a simple recipe for pancakes: Ingredients: 1 cup flour, 1 cup milk, 1 egg, 1 tbsp sugar, 1 tsp baking powder, 1/2 tsp salt. Instructions: 1. Mix all the dry ingredients together in a bowl. 2. Add the milk and egg and mix until smooth. 3. Heat a non-stick pan over medium heat. 4. Pour 1/4 cup of batter onto the pan. 5. Cook until bubbles form on the surface, then flip and cook for another minute. 6. Serve with your favorite toppings.",
    },
    {"role": "user", "content": "What are some popular toppings?"},
]

L = [messages_1, messages_2, messages_3, messages_4, messages_5, messages_6]

for i in range(len(L)):
    input_text = tokenizer.apply_chat_template(L[i], tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model_s.generate(
        inputs, max_new_tokens=200, top_p=TOP_P, do_sample=True, temperature=TEMPERATURE
    )
    with open(
        f"{BASE_PATH}/{CHECKPOINT.split('/')[-1]}_temp_{TEMPERATURE}_topp{TOP_P}_MT.txt",
        "a",
    ) as f:
        f.write("=" * 50 + "\n")
        f.write(tokenizer.decode(outputs[0]))
        f.write("\n")

print("🔥 Done!")

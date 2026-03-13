from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from src.agents.data_agent import process_document
from src.vision_model import run_vision_inference_simple
# from src.agents.explainer_agent import explain_with_ollama

# 1. Define the State
class AgenticState(TypedDict):
    file_path: str
    raw_text: str
    extracted_images: List[str]
    vision_insights: List[Dict[str, Any]]
    vision_output_images: List[str]
    text_insights: str
    final_report: str
    needs_web_search: bool
    conf_threshold: float
    anchor_priors: List[List[float]]


# 2. Define the Nodes (The Agents/Functions)
def data_ingestion_node(state: AgenticState):
    print(f"--- INGESTING FILE: {state['file_path']} ---")
    text, image_paths = process_document(state["file_path"])
    return {"raw_text": text, "extracted_images": image_paths}


def vision_processing_node(state: AgenticState):
    print("--- ROUTING IMAGES TO CUSTOM CBAM RESNET ---")
    insights = []
    output_images = []
    for img_path in state["extracted_images"]:
        insight = run_vision_inference_simple(
            img_path=img_path,
            conf_threshold=float(state.get("conf_threshold", 0.05)),
            anchor_priors=state.get("anchor_priors", [])
        )
        insights.append(insight)
        if insight.get("output_image"):
            output_images.append(insight["output_image"])
    return {"vision_insights": insights, "vision_output_images": output_images}


def text_processing_node(state: AgenticState):
    print("--- ROUTING TEXT TO LOCAL OLLAMA & PINECONE ---")
    # explanation = explain_with_ollama(state["raw_text"])
    explanation = "Simulated Ollama Explanation."
    return {"text_insights": explanation}


def compilation_node(state: AgenticState):
    print("--- COMPILING FINAL EXPLAINABLE REPORT ---")
    report = (
        f"Text Analysis:\n{state.get('text_insights')}\n\n"
        f"Vision Analysis:\n{state.get('vision_insights')}\n\n"
        f"Detection Output Images:\n{state.get('vision_output_images')}"
    )
    return {"final_report": report}


# 3. Build the Graph
workflow = StateGraph(AgenticState)

# Add Nodes
workflow.add_node("DataAgent", data_ingestion_node)
workflow.add_node("VisionAgent", vision_processing_node)
workflow.add_node("TextAgent", text_processing_node)
workflow.add_node("CompilerAgent", compilation_node)

# Add Edges (The Routing Logic)
workflow.set_entry_point("DataAgent")

# After parsing, we split the workflow to handle vision and text simultaneously
workflow.add_edge("DataAgent", "VisionAgent")
workflow.add_edge("DataAgent", "TextAgent")

# Once both are done, compile them
workflow.add_edge("VisionAgent", "CompilerAgent")
workflow.add_edge("TextAgent", "CompilerAgent")
workflow.add_edge("CompilerAgent", END)

# Compile the machine
app_router = workflow.compile()

# To run it locally:
if __name__ == "__main__":
    initial_state = {
        "file_path": "data/sample_warehouse_report.pdf",
        "raw_text": "",
        "extracted_images": [],
        "vision_insights": [],
        "vision_output_images": [],
        "text_insights": "",
        "final_report": "",
        "needs_web_search": False,
        "conf_threshold": 0.05,
        "anchor_priors": []
    }
    
    print("Initializing Multi-Agent Workflow...")
    result = app_router.invoke(initial_state)
    print("\nWorkflow Complete. Final Report Ready.")

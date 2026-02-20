"""
PREPTO - Automated Diagram Generator
Generates 50 SVG diagrams for engineering topics
Run time: ~20-30 minutes
"""

import os
import json
from openai import OpenAI
from pathlib import Path

print("="*70)
print("PREPTO DIAGRAM GENERATOR")
print("="*70)

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create directory structure
base_dir = Path("diagrams")
branches = ["cse", "ece", "mechanical", "electrical", "civil", "chemical"]

print("\n📁 Creating directory structure...")
for branch in branches:
    (base_dir / branch).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ diagrams/{branch}/")

# Define what diagrams to generate
diagram_plan = {
    "cse": {
        "binary_tree.svg": "Binary tree data structure with root, left child, right child nodes. Simple labeled diagram.",
        "stack_queue.svg": "Side-by-side comparison of Stack (LIFO) and Queue (FIFO) with push/pop operations.",
        "tcp_handshake.svg": "TCP 3-way handshake showing SYN, SYN-ACK, ACK between client and server.",
        "paging.svg": "Virtual memory paging diagram showing logical to physical address translation.",
        "normalization.svg": "Database normalization levels (1NF, 2NF, 3NF, BCNF) flow diagram.",
        "dfa_nfa.svg": "Simple DFA and NFA state diagram examples side by side.",
        "sorting_comparison.svg": "Time complexity comparison table for sorting algorithms (Bubble, Quick, Merge, Heap).",
        "bfs_dfs.svg": "Tree traversal comparison showing BFS (level-order) vs DFS (pre/in/post-order).",
        "heap_structure.svg": "Min-heap and Max-heap tree structures with example values.",
        "network_layers.svg": "OSI 7-layer and TCP/IP 4-layer model comparison.",
    },
    "ece": {
        "thevenin_norton.svg": "Thevenin and Norton equivalent circuit diagrams.",
        "bode_plot.svg": "Simple Bode plot showing magnitude and phase vs frequency.",
        "flip_flop_types.svg": "SR, D, JK, T flip-flop circuit symbols and truth tables.",
        "opamp_circuits.svg": "Inverting and non-inverting op-amp configurations.",
        "bjt_biasing.svg": "Common emitter BJT configuration with biasing resistors.",
        "am_fm_modulation.svg": "AM and FM waveforms showing carrier and modulated signals.",
        "transmission_line.svg": "Transmission line model with series impedance and shunt admittance.",
        "logic_gates.svg": "Basic logic gates (AND, OR, NOT, NAND, NOR, XOR) symbols.",
        "kmap_example.svg": "4-variable Karnaugh map example with grouping.",
        "filter_types.svg": "Low-pass, High-pass, Band-pass, Band-stop filter frequency responses.",
    },
    "mechanical": {
        "carnot_cycle.svg": "Carnot cycle P-V diagram with isothermal and adiabatic processes.",
        "rankine_cycle.svg": "Rankine cycle T-S diagram with pump, boiler, turbine, condenser.",
        "otto_diesel.svg": "Otto and Diesel cycle P-V diagrams comparison.",
        "bernoulli.svg": "Bernoulli's equation applied to pipe flow with varying diameter.",
        "venturi_meter.svg": "Venturi meter cross-section showing pressure measurement points.",
        "sfd_bmd.svg": "Simply supported beam with point load showing SFD and BMD diagrams.",
        "gear_train.svg": "Simple gear train with driver and driven gears showing speed ratio.",
        "governor.svg": "Centrifugal governor mechanism with rotating balls and sleeve.",
        "flywheel.svg": "Flywheel energy storage diagram showing torque fluctuation smoothing.",
        "stress_strain.svg": "Stress-strain curve showing elastic limit, yield point, ultimate strength.",
    },
    "electrical": {
        "transformer.svg": "Transformer construction showing primary, secondary windings and core.",
        "induction_motor.svg": "3-phase induction motor cross-section with stator and rotor.",
        "buck_boost.svg": "Buck-boost converter circuit diagram with switch and diode.",
        "inverter.svg": "Single-phase inverter circuit with switching elements.",
        "laplace_transform.svg": "Common Laplace transform pairs table.",
        "wheatstone_bridge.svg": "Wheatstone bridge circuit for resistance measurement.",
        "fault_analysis.svg": "Single-line diagram showing symmetrical fault analysis.",
        "power_system.svg": "Simple power system single-line diagram with generator, transformer, load.",
        "thyristor.svg": "Thyristor (SCR) V-I characteristics curve.",
        "synchronous_machine.svg": "Synchronous generator phasor diagram.",
    },
    "civil": {
        "truss_structure.svg": "Warren truss structure with labeled members and joints.",
        "soil_classification.svg": "Soil classification triangle (sand, silt, clay percentages).",
        "water_treatment.svg": "Water treatment process flow: screening, sedimentation, filtration, disinfection.",
        "highway_cross_section.svg": "Highway cross-section showing carriageway, shoulder, median.",
        "beam_loading.svg": "Simply supported beam with UDL and point loads.",
    },
    "chemical": {
        "distillation_column.svg": "Distillation column with feed, overhead, bottoms streams.",
        "heat_exchanger.svg": "Shell and tube heat exchanger cross-section.",
        "pfr_cstr.svg": "PFR (Plug Flow Reactor) and CSTR comparison diagrams.",
        "pid_controller.svg": "PID controller block diagram with feedback loop.",
        "mass_transfer.svg": "Binary distillation x-y diagram for McCabe-Thiele method.",
    }
}

total_diagrams = sum(len(files) for files in diagram_plan.values())
print(f"\n📊 Plan: Generate {total_diagrams} SVG diagrams")

generated_count = 0

print("\n🚀 Starting generation...\n")

for branch, diagrams in diagram_plan.items():
    print(f"{'='*70}")
    print(f"📝 Branch: {branch.upper()}")
    print(f"🎯 Diagrams: {len(diagrams)}")
    
    for filename, description in diagrams.items():
        filepath = base_dir / branch / filename
        
        prompt = f"""Generate a simple, clean SVG diagram for engineering education.

Topic: {description}

Requirements:
- SIMPLE line diagram (not realistic/photographic)
- Clean, minimal design
- Black lines on white background
- Include essential labels
- Educational, easy to understand
- Suitable for web display
- Size: 600x400 pixels viewBox

Return ONLY the complete SVG code (starting with <svg> and ending with </svg>).
No explanation, no markdown backticks, just the SVG code."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # GPT-4o is better at SVG generation
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating simple, educational SVG diagrams for engineering concepts. Generate clean, minimalist line diagrams."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower for consistency
            )
            
            svg_content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if svg_content.startswith("```"):
                svg_content = svg_content.split("```")[1]
                if svg_content.startswith("svg"):
                    svg_content = svg_content[3:]
                svg_content = svg_content.strip()
            
            # Ensure it starts with <svg
            if not svg_content.startswith("<svg"):
                print(f"  ✗ {filename}: Invalid SVG generated")
                continue
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            generated_count += 1
            print(f"  ✓ {filename} ({generated_count}/{total_diagrams})")
            
        except Exception as e:
            print(f"  ✗ {filename}: Error - {str(e)[:60]}")
    
    print(f"✅ {branch.upper()} complete\n")

print("="*70)
print(f"✅ GENERATION COMPLETE!")
print(f"📊 Generated: {generated_count}/{total_diagrams} diagrams")
print("="*70)

# Create summary
print("\n📋 Generated Files:")
for branch in branches:
    files = list((base_dir / branch).glob("*.svg"))
    print(f"\n{branch.upper()}: {len(files)} diagrams")
    for f in sorted(files):
        print(f"  ✓ {f.name}")

print("\n" + "="*70)
print("🎉 SUCCESS!")
print("="*70)
print(f"📁 Location: ./diagrams/")
print(f"📊 Total: {generated_count} SVG files")
print("\n🚀 Next steps:")
print("1. Review diagrams (open in browser)")
print("2. Upload entire 'diagrams' folder to GitHub")
print("3. Deploy to Prepto")
print("="*70)

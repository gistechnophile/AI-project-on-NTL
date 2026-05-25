"""Generate a presentation-ready flowchart mapping course sessions to project applications."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Use the venv Python
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0

fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Title
ax.text(10, 13.5, 'AI & Large Models Course → NTL Population Estimation Project',
        fontsize=22, fontweight='bold', ha='center', va='center', color='#1a1a2e')
ax.text(10, 13.0, 'Mapping Every Session to Production Implementation',
        fontsize=14, ha='center', va='center', color='#4a4a6a')

# Color scheme
colors = {
    'foundation': '#e8f4f8',      # light blue
    'foundation_border': '#2196f3',
    'data': '#f3e5f5',            # light purple
    'data_border': '#9c27b0',
    'model': '#e8f5e9',           # light green
    'model_border': '#4caf50',
    'rag_agent': '#fff3e0',       # light orange
    'rag_agent_border': '#ff9800',
    'multimodal': '#fce4ec',      # light pink
    'multimodal_border': '#e91e63',
    'arrow': '#607d8b',
}

# Layout: 4 rows, each row has session boxes on left, project component on right
row_y_positions = [11.2, 8.8, 6.4, 4.0, 1.6]
row_heights = 1.8

sessions_data = [
    {
        'group': 'FOUNDATION & WORKFLOW',
        'group_color': colors['foundation_border'],
        'items': [
            ('S1: AI Flywheel', 'Encounter→Decompose→Simulate→Critique\nApplied to NTL saturation problem'),
            ('S2: Prompting', 'Structured RAG prompts\nIterative paper refinement'),
            ('S3: ML Foundations', 'ResNet-18 + backprop\nModel selection framework'),
        ]
    },
    {
        'group': 'DATA & COMPUTE',
        'group_color': colors['data_border'],
        'items': [
            ('S4: Data Thinking', 'Nodata handling, MNAR missingness\nQuality audit (5 dimensions)'),
            ('S5: Compute Reality', 'Batch_size=8 trade-off\n137MB checkpoint, 8GB VRAM budget'),
        ]
    },
    {
        'group': 'MODELS & ALIGNMENT',
        'group_color': colors['model_border'],
        'items': [
            ('S6: Transformers', 'TF-IDF tokenization in RAG\n1D conv chosen over attention (efficiency)'),
            ('S7: Alignment', 'Hard clamp [-2,16] as guardrail\nHuber loss robustness alignment'),
        ]
    },
    {
        'group': 'RAG & AGENTS',
        'group_color': colors['rag_agent_border'],
        'items': [
            ('S8: RAG', 'TF-IDF retrieval from 8 papers\nLiterature-grounded report generation'),
            ('S9: Agents', 'ReAct debugging loop\nTool use: Shell/ReadFile/WriteFile'),
        ]
    },
    {
        'group': 'MULTIMODAL & ENGINEERING',
        'group_color': colors['multimodal_border'],
        'items': [
            ('S10: Multimodal AI', '4-channel fusion: NTL+POP+Surface+Volume\nJoint reasoning in single forward pass'),
            ('S11: Claude Code', '100+ agentic interactions\nGit history as lab notebook (65+ files)'),
            ('S12: Token Economics', 'ResNet-18 serving efficiency\nTF-IDF over SBERT (cost optimization)'),
        ]
    },
]

def draw_rounded_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=9, fontweight='normal', text_color='#1a1a2e'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, fontsize=fontsize, fontweight=fontweight,
            ha='center', va='center', color=text_color, wrap=True,
            linespacing=1.3)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=colors['arrow']):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                               connectionstyle='arc3,rad=0'))

# Draw rows
for row_idx, row_data in enumerate(sessions_data):
    y = row_y_positions[row_idx]
    items = row_data['items']
    n_items = len(items)
    group_color = row_data['group_color']
    
    # Group label on far left
    ax.text(0.3, y + row_heights/2, row_data['group'],
            fontsize=11, fontweight='bold', ha='left', va='center',
            color=group_color, rotation=90,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=group_color, linewidth=2))
    
    # Calculate widths for session boxes
    session_w = 3.8
    gap = 0.4
    start_x = 1.8
    
    for item_idx, (session_title, app_text) in enumerate(items):
        x = start_x + item_idx * (session_w + gap)
        
        # Session box (left side of pair)
        draw_rounded_box(ax, x, y + 0.3, session_w, row_heights - 0.6,
                        session_title, '#ffffff', group_color, fontsize=10, fontweight='bold')
        
        # Application box (right side)
        app_x = x + session_w + 0.3
        app_w = 8.5
        draw_rounded_box(ax, app_x, y + 0.3, app_w, row_heights - 0.6,
                        app_text, group_color, group_color, fontsize=9, text_color='#1a1a2e')
        
        # Arrow from session to application
        draw_arrow(ax, x + session_w, y + row_heights/2, app_x, y + row_heights/2, group_color)

# Add bottom legend
legend_y = 0.5
ax.text(10, legend_y, 
        'Every session concept was implemented in production code — no theory left unused',
        fontsize=12, ha='center', va='center', color='#555555', style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#cccccc', linewidth=1))

# Add project outcome banner at top right
banner_x, banner_y = 16.5, 12.3
banner_w, banner_h = 3.0, 0.8
draw_rounded_box(ax, banner_x, banner_y, banner_w, banner_h,
                'R = 0.881\nMAE = 2.24', '#1a1a2e', '#1a1a2e',
                fontsize=12, fontweight='bold', text_color='white')

plt.tight_layout()
plt.savefig('E:\\Private\\Lectures\\AI\\AI Application\\paklight-pop\\outputs\\course_materials_flowchart.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Flowchart saved to: outputs/course_materials_flowchart.png")

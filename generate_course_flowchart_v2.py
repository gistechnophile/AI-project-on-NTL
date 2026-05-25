"""Generate a cleaner, presentation-ready flowchart."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(1, 1, figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis('off')

# Title
ax.text(11, 15.5, 'AI & Large Models Course  →  NTL Population Estimation Project',
        fontsize=24, fontweight='bold', ha='center', va='center', color='#1a1a2e')
ax.text(11, 15.0, 'How Every Session Was Applied to Production Code',
        fontsize=14, ha='center', va='center', color='#555555')

# Result badge
badge = FancyBboxPatch((18.5, 14.6), 3.0, 0.7, boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor='#1a1a2e', edgecolor='none')
ax.add_patch(badge)
ax.text(20.0, 14.95, 'R = 0.881  |  MAE = 2.24', fontsize=12, fontweight='bold',
        ha='center', va='center', color='white')

# Colors
colors = {
    'foundation': '#2196f3',
    'data': '#9c27b0',
    'model': '#4caf50',
    'rag_agent': '#ff9800',
    'multimodal': '#e91e63',
    'bg_light': '#f5f5f5',
}

# Data: (group_label, group_color, sessions_list)
# Each session: (session_num, session_title, application)
rows = [
    ('FOUNDATION & WORKFLOW', colors['foundation'], [
        ('S1', 'AI Flywheel', 'Encounter → Decompose → Simulate → Critique\napplied to NTL saturation problem'),
        ('S2', 'Prompting', 'Structured RAG prompts\nIterative paper refinement'),
        ('S3', 'ML Foundations', 'ResNet-18 + backpropagation\nModel selection framework'),
    ]),
    ('DATA & COMPUTE', colors['data'], [
        ('S4', 'Data Thinking', 'Nodata handling (65535, 4294967295)\nMNAR missingness, 5-dimension quality audit'),
        ('S5', 'Compute Reality', 'Batch_size=8 trade-off\n137MB checkpoint, 8GB VRAM budget'),
    ]),
    ('MODELS & ALIGNMENT', colors['model'], [
        ('S6', 'Transformers', 'TF-IDF tokenization in RAG engine\n1D conv over attention (efficiency)'),
        ('S7', 'Alignment', 'Hard clamp [-2, 16] as safety guardrail\nHuber loss robustness alignment'),
    ]),
    ('RAG & AGENTS', colors['rag_agent'], [
        ('S8', 'RAG', 'TF-IDF retrieval from 8 papers (192 chunks)\nLiterature-grounded report generation'),
        ('S9', 'Agents', 'ReAct debugging loop\nTool use: Shell / ReadFile / WriteFile'),
    ]),
    ('MULTIMODAL & ENGINEERING', colors['multimodal'], [
        ('S10', 'Multimodal AI', '4-channel fusion: NTL + POP + Surface + Volume\nJoint reasoning in single forward pass'),
        ('S11', 'Claude Code', '100+ agentic interactions\nGit history as lab notebook (65+ files)'),
        ('S12', 'Token Economics', 'ResNet-18 serving efficiency\nTF-IDF over SBERT (cost optimization)'),
    ]),
]

y_start = 13.2
row_height = 2.2
row_gap = 0.4

def draw_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=9, fontweight='normal', textcolor='#1a1a2e'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, fontsize=fontsize, fontweight=fontweight,
            ha='center', va='center', color=textcolor, wrap=True, linespacing=1.25)

for row_idx, (group_label, group_color, sessions) in enumerate(rows):
    y = y_start - row_idx * (row_height + row_gap)
    n = len(sessions)
    
    # Group label banner
    banner_h = row_height + 0.3
    banner = FancyBboxPatch((0.2, y - 0.15), 1.6, banner_h,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=group_color, edgecolor='none')
    ax.add_patch(banner)
    ax.text(1.0, y + banner_h/2 - 0.15, group_label, fontsize=10, fontweight='bold',
            ha='center', va='center', color='white', rotation=90, linespacing=0.9)
    
    # Session boxes
    box_w = 4.2
    app_w = 6.8
    gap = 0.35
    start_x = 2.2
    
    for i, (snum, stitle, app) in enumerate(sessions):
        x = start_x + i * (box_w + gap + app_w + gap)
        
        # Session number + title box
        session_text = f'{snum}\n{stitle}'
        draw_box(ax, x, y, box_w, row_height, session_text, 'white', group_color,
                 fontsize=10, fontweight='bold')
        
        # Application box
        app_x = x + box_w + gap
        draw_box(ax, app_x, y, app_w, row_height, app, '#fafafa', group_color,
                 fontsize=9, textcolor='#333333')
        
        # Arrow
        ax.annotate('', xy=(app_x, y + row_height/2), xytext=(x + box_w, y + row_height/2),
                    arrowprops=dict(arrowstyle='->', color=group_color, lw=2.5,
                                   connectionstyle='arc3,rad=0'))

# Footer
footer_y = 0.6
ax.plot([1, 21], [footer_y + 0.4, footer_y + 0.4], color='#cccccc', lw=1)
ax.text(11, footer_y, 'Every session concept was implemented in production code — no theory was left unused',
        fontsize=12, ha='center', va='center', color='#666666', style='italic')

plt.tight_layout()
plt.savefig('E:\\Private\\Lectures\\AI\\AI Application\\paklight-pop\\outputs\\course_flowchart_presentation.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Presentation flowchart saved!")

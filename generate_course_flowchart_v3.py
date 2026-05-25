"""Generate a clean, complete flowchart where all 12 sessions fit perfectly."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(1, 1, figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(10, 11.5, 'AI & Large Models Course  →  NTL Population Estimation Project',
        fontsize=20, fontweight='bold', ha='center', va='center', color='#1a1a2e')
ax.text(10, 11.1, 'How Every Session Was Applied to Production Code',
        fontsize=12, ha='center', va='center', color='#555555')

# Result badge
badge = FancyBboxPatch((16.8, 10.7), 2.8, 0.55, boxstyle="round,pad=0.02,rounding_size=0.12",
                       facecolor='#1a1a2e', edgecolor='none')
ax.add_patch(badge)
ax.text(18.2, 10.98, 'R = 0.881  |  MAE = 2.24', fontsize=11, fontweight='bold',
        ha='center', va='center', color='white')

# Colors
colors = {
    'foundation': '#2196f3',
    'data': '#9c27b0',
    'model': '#4caf50',
    'rag_agent': '#ff9800',
    'multimodal': '#e91e63',
}

# Layout constants
left_margin = 0.3
label_w = 1.4
session_w = 2.6
app_w = 4.8
gap = 0.25
arrow_len = 0.35
row_h = 1.55
row_gap = 0.35
start_y = 9.8

rows = [
    ('FOUNDATION &\nWORKFLOW', colors['foundation'], [
        ('S1', 'AI Flywheel', 'Encounter→Decompose→\nSimulate→Critique'),
        ('S2', 'Prompting', 'Structured RAG prompts\nIterative refinement'),
        ('S3', 'ML Foundations', 'ResNet-18 + backprop\nModel selection'),
    ]),
    ('DATA &\nCOMPUTE', colors['data'], [
        ('S4', 'Data Thinking', 'Nodata handling\nMNAR missingness'),
        ('S5', 'Compute Reality', 'Batch_size=8 trade-off\n8GB VRAM budget'),
    ]),
    ('MODELS &\nALIGNMENT', colors['model'], [
        ('S6', 'Transformers', 'TF-IDF tokenization\n1D conv efficiency'),
        ('S7', 'Alignment', 'Hard clamp guardrail\nHuber robustness'),
    ]),
    ('RAG &\nAGENTS', colors['rag_agent'], [
        ('S8', 'RAG', 'TF-IDF from 8 papers\nLiterature reports'),
        ('S9', 'Agents', 'ReAct debugging loop\nTool use (Shell/R/W)'),
    ]),
    ('MULTIMODAL &\nENGINEERING', colors['multimodal'], [
        ('S10', 'Multimodal AI', '4-channel fusion\nJoint reasoning'),
        ('S11', 'Claude Code', '100+ agentic interactions\nGit lab notebook'),
        ('S12', 'Token Economics', 'ResNet-18 efficiency\nTF-IDF over SBERT'),
    ]),
]

def draw_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=8.5, fw='normal', tc='#1a1a2e'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, fontsize=fontsize, fontweight=fw,
            ha='center', va='center', color=tc, linespacing=1.15)

for row_idx, (group_label, group_color, sessions) in enumerate(rows):
    y = start_y - row_idx * (row_h + row_gap)
    n = len(sessions)
    
    # Group label
    banner = FancyBboxPatch((left_margin, y - 0.1), label_w, row_h + 0.2,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=group_color, edgecolor='none')
    ax.add_patch(banner)
    ax.text(left_margin + label_w/2, y + row_h/2, group_label, fontsize=9, fontweight='bold',
            ha='center', va='center', color='white', linespacing=0.95)
    
    # Session + app pairs
    base_x = left_margin + label_w + 0.4
    for i, (snum, stitle, app) in enumerate(sessions):
        x = base_x + i * (session_w + arrow_len + app_w + gap)
        
        # Session box
        session_text = f'{snum}\n{stitle}'
        draw_box(ax, x, y, session_w, row_h, session_text, 'white', group_color,
                 fontsize=9.5, fw='bold')
        
        # Arrow
        ax.annotate('', xy=(x + session_w + arrow_len, y + row_h/2),
                    xytext=(x + session_w, y + row_h/2),
                    arrowprops=dict(arrowstyle='->', color=group_color, lw=2))
        
        # Application box
        app_x = x + session_w + arrow_len
        draw_box(ax, app_x, y, app_w, row_h, app, '#fafafa', group_color, fontsize=8.5)

# Footer
ax.plot([1, 19], [0.45, 0.45], color='#cccccc', lw=1)
ax.text(10, 0.25, 'Every session concept was implemented in production code — no theory was left unused',
        fontsize=11, ha='center', va='center', color='#555555', style='italic')

plt.tight_layout()
plt.savefig('E:\\Private\\Lectures\\AI\\AI Application\\paklight-pop\\outputs\\course_flowchart_final.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Final flowchart saved!')

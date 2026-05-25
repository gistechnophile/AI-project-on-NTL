"""Complete flowchart — all 12 sessions fully visible."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# WIDE canvas to fit all 3-session rows
fig, ax = plt.subplots(1, 1, figsize=(24, 13))
ax.set_xlim(0, 24)
ax.set_ylim(0, 13)
ax.axis('off')

# Title
ax.text(12, 12.4, 'AI & Large Models Course  →  NTL Population Estimation Project',
        fontsize=22, fontweight='bold', ha='center', va='center', color='#1a1a2e')
ax.text(12, 11.95, 'How Every Session Was Applied to Production Code',
        fontsize=13, ha='center', va='center', color='#555555')

# Result badge
badge = FancyBboxPatch((20.5, 11.5), 3.0, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                       facecolor='#1a1a2e', edgecolor='none')
ax.add_patch(badge)
ax.text(22.0, 11.8, 'R = 0.881  |  MAE = 2.24', fontsize=12, fontweight='bold',
        ha='center', va='center', color='white')

# Colors
colors = {
    'foundation': '#2196f3',
    'data': '#9c27b0',
    'model': '#4caf50',
    'rag_agent': '#ff9800',
    'multimodal': '#e91e63',
}

# Layout
left_margin = 0.3
label_w = 1.6
session_w = 3.0
app_w = 5.5
gap = 0.3
arrow_len = 0.3
row_h = 1.45
row_gap = 0.3
start_y = 10.6

rows = [
    ('FOUNDATION & WORKFLOW', colors['foundation'], [
        ('S1', 'AI Flywheel', 'Encounter→Decompose→Simulate→Critique\napplied to NTL saturation'),
        ('S2', 'Prompting', 'Structured RAG prompts\nIterative paper refinement'),
        ('S3', 'ML Foundations', 'ResNet-18 + backpropagation\nModel selection framework'),
    ]),
    ('DATA & COMPUTE', colors['data'], [
        ('S4', 'Data Thinking', 'Nodata handling (65535, 4294967295)\nMNAR missingness, quality audit'),
        ('S5', 'Compute Reality', 'Batch_size=8 trade-off\n137MB checkpoint, 8GB VRAM'),
    ]),
    ('MODELS & ALIGNMENT', colors['model'], [
        ('S6', 'Transformers', 'TF-IDF tokenization in RAG\n1D conv over attention (efficiency)'),
        ('S7', 'Alignment', 'Hard clamp [-2,16] as safety guardrail\nHuber loss robustness alignment'),
    ]),
    ('RAG & AGENTS', colors['rag_agent'], [
        ('S8', 'RAG', 'TF-IDF retrieval from 8 papers (192 chunks)\nLiterature-grounded report generation'),
        ('S9', 'Agents', 'ReAct debugging loop\nTool use: Shell / ReadFile / WriteFile'),
    ]),
    ('MULTIMODAL & ENGINEERING', colors['multimodal'], [
        ('S10', 'Multimodal AI', '4-channel fusion: NTL+POP+Surface+Volume\nJoint reasoning in single forward pass'),
        ('S11', 'Claude Code', '100+ agentic interactions\nGit history as lab notebook (65+ files)'),
        ('S12', 'Token Economics', 'ResNet-18 serving efficiency\nTF-IDF over SBERT (cost opt)'),
    ]),
]

def draw_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=9, fw='normal', tc='#1a1a2e'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, fontsize=fontsize, fontweight=fw,
            ha='center', va='center', color=tc, linespacing=1.1)

for row_idx, (group_label, group_color, sessions) in enumerate(rows):
    y = start_y - row_idx * (row_h + row_gap)
    n = len(sessions)
    
    # Group label
    banner = FancyBboxPatch((left_margin, y - 0.08), label_w, row_h + 0.16,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            facecolor=group_color, edgecolor='none')
    ax.add_patch(banner)
    ax.text(left_margin + label_w/2, y + row_h/2, group_label, fontsize=9, fontweight='bold',
            ha='center', va='center', color='white', linespacing=0.9)
    
    # Calculate starting X so all sessions fit within x=0 to x=24
    # Total width per session = session_w + arrow_len + app_w + gap
    unit_w = session_w + arrow_len + app_w + gap
    total_content_w = n * unit_w - gap  # remove last gap
    base_x = left_margin + label_w + 0.5
    
    for i, (snum, stitle, app) in enumerate(sessions):
        x = base_x + i * unit_w
        
        # Session box
        session_text = f'{snum}: {stitle}'
        draw_box(ax, x, y, session_w, row_h, session_text, 'white', group_color,
                 fontsize=10, fw='bold')
        
        # Arrow
        ax.annotate('', xy=(x + session_w + arrow_len, y + row_h/2),
                    xytext=(x + session_w, y + row_h/2),
                    arrowprops=dict(arrowstyle='->', color=group_color, lw=2.5))
        
        # Application box
        app_x = x + session_w + arrow_len
        draw_box(ax, app_x, y, app_w, row_h, app, '#f8f9fa', group_color, fontsize=9)

# Footer
ax.plot([1, 23], [0.4, 0.4], color='#cccccc', lw=1)
ax.text(12, 0.2, 'Every session concept was implemented in production code — no theory was left unused',
        fontsize=11, ha='center', va='center', color='#555555', style='italic')

plt.tight_layout()
plt.savefig('E:\\Private\\Lectures\\AI\\AI Application\\paklight-pop\\outputs\\course_flowchart_complete.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Complete flowchart saved!')

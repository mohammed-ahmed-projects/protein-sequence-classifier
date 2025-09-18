# app.py        
import uuid  
import os  
import threading, time  
import gradio as gr  
import pandas as pd  
import matplotlib.pyplot as plt  
from io import BytesIO  
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet  
from reportlab.lib.pagesizes import A4  
from reportlab.lib import colors
  
from models import ProteinClassifier  

print("Initializing Protein Classifier...")
classifier = ProteinClassifier()  
print("Protein Classifier initialized successfully!")

# CSS with subtle improvements and dark mode support
CUSTOM_CSS = """
.gradio-container { 
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

/* Headers */
.gr-markdown h1 {
    text-align: center !important;
    font-weight: 600 !important;
    font-size: 2.2em !important;
    margin-bottom: 0.5em !important;
    border-bottom: 2px solid #27ae60 !important;
    padding-bottom: 0.3em !important;
}

/* Buttons */
.gr-button-primary {
    background: #27ae60 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    transition: background-color 0.2s ease !important;
}

.gr-button-primary:hover {
    background: #229954 !important;
}

/* Input containers */
.gr-box {
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

/* Textboxes */
.gr-textbox textarea {
    border-radius: 6px !important;
    font-family: 'Consolas', 'Monaco', monospace !important;
    font-size: 14px !important;
}

.gr-textbox textarea:focus {
    border-color: #27ae60 !important;
    box-shadow: 0 0 0 2px rgba(39, 174, 96, 0.2) !important;
}

/* Dropdowns */
.gr-dropdown select {
    border-radius: 6px !important;
}

/* Labels */
.gr-label {
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* Result labels */
.gr-label-output {
    border-radius: 8px !important;
    padding: 16px !important;
    font-weight: 600 !important;
    text-align: center !important;
}

/* Images */
.gr-image {
    border-radius: 8px !important;
}

/* File components */
.gr-file {
    border-radius: 8px !important;
}

/* Row spacing */
.gr-row {
    margin-bottom: 16px !important;
}
"""
  
# File auto-delete  
DELETE_DELAY = 15 * 60  
def schedule_delete(path):  
    def delete_file():  
        time.sleep(DELETE_DELAY)  
        if os.path.exists(path):  
            try:
                os.remove(path)  
                print(f"Auto-deleted: {path}")
            except Exception as e:
                print(f"Failed to delete {path}: {e}")
    threading.Thread(target=delete_file, daemon=True).start()  
  
def save_csv(analysis_dict, filename="analysis.csv"):  
    try:
        if isinstance(analysis_dict, dict) and "Error" in analysis_dict:
            df = pd.DataFrame([{"Status": "Error", "Message": analysis_dict["Error"]}])
        else:
            df = pd.DataFrame([analysis_dict])  
        df.to_csv(filename, index=False)  
        schedule_delete(filename)  
        return filename  
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return None
  
def save_pdf(analysis_dict, filename="analysis.pdf"):  
    try:
        doc = SimpleDocTemplate(filename, pagesize=A4)  
        styles = getSampleStyleSheet()  
        elements = [
            Paragraph("Protein Sequence Analysis Report", styles["Title"]), 
            Spacer(1, 12)
        ]  
  
        if isinstance(analysis_dict, dict):  
            if "Error" in analysis_dict:
                elements.append(Paragraph(f"Error: {analysis_dict['Error']}", styles["Normal"]))
                data = [["Status", "Error"], ["Message", analysis_dict["Error"]]]
            else:
                data = [[k, str(v)] for k, v in analysis_dict.items()]  
        else:  
            data = [["Analysis", str(analysis_dict)]]  
  
        table = Table([["Property", "Value"]] + data)  
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)  
  
        doc.build(elements)  
        schedule_delete(filename)  
        return filename  
    except Exception as e:
        print(f"Error saving PDF: {e}")
        return None
  
def make_prob_chart(prob):  
    try:
        fig, ax = plt.subplots(figsize=(8, 5))  
        
        bars = ax.bar(["Non-enzyme", "Enzyme"], [1 - prob, prob], 
                     color=['#e74c3c', '#27ae60'], alpha=0.8, 
                     edgecolor='white', linewidth=2)
        
        ax.set_ylim(0, 1)  
        ax.set_ylabel("Probability", fontsize=12, fontweight='500')  
        ax.set_title("Classification Probability", fontsize=14, fontweight='600')  
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.1%}', ha='center', va='bottom', 
                   fontsize=11, fontweight='500')
        
        # Clean styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        buf = BytesIO()  
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", 
                   facecolor='white')  
        plt.close(fig)  
        buf.seek(0)  
        return buf.getvalue()  
    except Exception as e:
        print(f"Error creating probability chart: {e}")
        return None
  
def predict_protein(sequence, model_type):  
    print(f"Processing sequence: '{sequence[:50]}...' with model: {model_type}")
    
    try:  
        if not sequence or sequence.strip() == "":
            error_msg = "Please enter a protein sequence"
            return (  
                "No Input",  
                "0.000",  
                error_msg,  
                None,  
                None,  
                None,  
                None,  
                f"<div style='padding:10px; background-color:#f8d7da; border:1px solid #f5c6cb; border-radius:5px; color:#721c24;'>{error_msg}</div>",  
            )
        
        result, confidence, analysis, plot_buf = classifier.predict_sequence(sequence.strip(), model_type)  
        conf_val = float(confidence) if isinstance(confidence, (int, float)) else 0.0
  
        if isinstance(analysis, dict) and "Error" in analysis:  
            error_msg = analysis["Error"]
            return (  
                result,  
                f"{conf_val:.3f}",  
                error_msg,  
                None,  
                None,  
                None,  
                None,  
                f"<div style='padding:10px; background-color:#f8d7da; border:1px solid #f5c6cb; border-radius:5px; color:#721c24;'>Error: {error_msg}</div>",  
            )  
  
        comp_plot = None
        output_file = None
        if plot_buf:  
            try:
                comp_plot = plot_buf.getvalue()  
                unique_name = f"composition_{uuid.uuid4().hex[:8]}.png"  
                with open(unique_name, "wb") as f:  
                    f.write(comp_plot)  
                output_file = unique_name  
                schedule_delete(output_file)  
            except Exception as e:
                print(f"Error saving composition plot: {e}")
  
        prob_chart = make_prob_chart(conf_val)  
        csv_file = save_csv(analysis, f"report_{uuid.uuid4().hex[:8]}.csv")  
        pdf_file = save_pdf(analysis, f"report_{uuid.uuid4().hex[:8]}.pdf")  
  
        if hasattr(classifier, 'models_loaded') and not classifier.models_loaded.get(model_type, False):
            warning_html = "<div style='padding:10px; background-color:#fff3cd; border:1px solid #ffeeba; border-radius:5px; color:#856404;'>Warning: Using untrained model - results may not be accurate.</div>"
        else:
            warning_html = "<div style='padding:10px; background-color:#d4edda; border:1px solid #c3e6cb; border-radius:5px; color:#155724;'>Analysis completed. Files will auto-delete after 15 minutes.</div>"  
  
        return (  
            result,  
            f"{conf_val:.3f}",  
            str(analysis) if not isinstance(analysis, dict) else "\n".join([f"{k}: {v}" for k, v in analysis.items()]),
            comp_plot,  
            prob_chart,  
            csv_file,  
            pdf_file,  
            warning_html,  
        )  
  
    except Exception as e:  
        error_msg = f"Error: {str(e)}"
        print(f"Error in predict_protein: {e}")
        return (  
            "Error",  
            "0.000",  
            error_msg,  
            None,  
            None,  
            None,  
            None,  
            f"<div style='padding:10px; background-color:#f8d7da; border:1px solid #f5c6cb; border-radius:5px; color:#721c24;'>{error_msg}</div>",  
        )  

# Sample sequences
SAMPLE_SEQUENCES = {
    "Short Test": "MKLASVGH",
    "Lysozyme (Enzyme)": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "Insulin (Non-enzyme)": "GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
}

# Gradio interface  
with gr.Blocks(theme=gr.themes.Soft(), title="Protein Sequence Classifier", css=CUSTOM_CSS) as demo:  
    gr.Markdown("""
    # Protein Sequence Classifier
    
    Enter a protein sequence to classify it as enzyme or non-enzyme.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            seq_input = gr.Textbox(
                lines=5, 
                label="Protein Sequence", 
                placeholder="Enter protein sequence using single letter amino acid codes...\nExample: MKLASVGH",
                info="Supported amino acids: A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y"
            )
            
            sample_dropdown = gr.Dropdown(
                choices=list(SAMPLE_SEQUENCES.keys()),
                label="Sample Sequences",
                value=None,
                info="Select a pre-loaded example"
            )
            
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                ["CNN", "LSTM"], 
                value="CNN", 
                label="Model Type",
                info="Choose neural network architecture"
            )
            run_btn = gr.Button("Run Analysis", variant="primary", size="lg")
    
    def load_sample(sample_name):
        if sample_name and sample_name in SAMPLE_SEQUENCES:
            return SAMPLE_SEQUENCES[sample_name]
        return ""
    
    sample_dropdown.change(fn=load_sample, inputs=[sample_dropdown], outputs=[seq_input])
    
    gr.Markdown("---")
    
    with gr.Row():  
        result_out = gr.Label(label="Prediction", container=True)  
        conf_out = gr.Label(label="Confidence Score", container=True)  
  
    analysis_out = gr.Textbox(
        label="Detailed Analysis", 
        lines=8, 
        max_lines=15,
        show_copy_button=True
    )
    
    with gr.Row():  
        plot_out = gr.Image(label="Amino Acid Composition", container=True)  
        prob_out = gr.Image(label="Probability Chart", container=True)  
  
    with gr.Row():  
        file_csv = gr.File(label="CSV Report")  
        file_pdf = gr.File(label="PDF Report")  
  
    warning_out = gr.HTML()  
  
    run_btn.click(  
        fn=predict_protein,  
        inputs=[seq_input, model_choice],  
        outputs=[result_out, conf_out, analysis_out, plot_out, prob_out, file_csv, file_pdf, warning_out],  
    )
    
    gr.Markdown("""
    ---
    ### Technical Details:
    - **CNN**: Convolutional Neural Network for local sequence patterns
    - **LSTM**: Long Short-Term Memory for sequential dependencies
    - Analysis includes: molecular weight, isoelectric point, charge, hydrophobicity
    - Files automatically deleted after 15 minutes
    """)
  
if __name__ == "__main__":  
    print("Starting application...")
    demo.launch(
        show_api=False,
        server_name="0.0.0.0",
        server_port=7860
            )

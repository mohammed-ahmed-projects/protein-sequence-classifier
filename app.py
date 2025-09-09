# app.py        
import uuid  
import os  
import threading, time  
import gradio as gr  
import pandas as pd  
import matplotlib.pyplot as plt  
from io import BytesIO  
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table  
from reportlab.lib.styles import getSampleStyleSheet  
from reportlab.lib.pagesizes import A4  
  
from models import ProteinClassifier  
  
classifier = ProteinClassifier()  
  
# File auto-delete  
DELETE_DELAY = 15 * 60  # 15 minutes  
def schedule_delete(path):  
    def delete_file():  
        time.sleep(DELETE_DELAY)  
        if os.path.exists(path):  
            os.remove(path)  
    threading.Thread(target=delete_file, daemon=True).start()  
  
# --- Helpers for exporting results ---  
def save_csv(analysis_dict, filename="analysis.csv"):  
    df = pd.DataFrame([analysis_dict])  
    df.to_csv(filename, index=False)  
    schedule_delete(filename)  
    return filename  
  
def save_pdf(analysis_dict, filename="analysis.pdf"):  
    doc = SimpleDocTemplate(filename, pagesize=A4)  
    styles = getSampleStyleSheet()  
    elements = [Paragraph("Protein Sequence Analysis Report", styles["Title"]), Spacer(1, 12)]  
  
    if isinstance(analysis_dict, dict):  
        data = [[k, str(v)] for k, v in analysis_dict.items()]  
    else:  
        data = [["Analysis", str(analysis_dict)]]  
  
    table = Table([["Property", "Value"]] + data)  
    elements.append(table)  
  
    doc.build(elements)  
    schedule_delete(filename)  
    return filename  
  
def make_prob_chart(prob):  
    fig, ax = plt.subplots()  
    ax.bar(["Non-enzyme", "Enzyme"], [1 - prob, prob], color=["red", "green"])  
    ax.set_ylim(0, 1)  
    ax.set_ylabel("Probability")  
    ax.set_title("Classification Probability")  
    buf = BytesIO()  
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")  
    plt.close(fig)  
    buf.seek(0)  
    return buf.getvalue()  
  
# --- Main prediction function ---  
def predict_protein(sequence, model_type):  
    try:  
        result, confidence, analysis, plot_buf = classifier.predict_sequence(sequence, model_type)  
        conf_val = float(confidence)  
  
        # if error in analysis, don’t create files  
        if isinstance(analysis, dict) and "Error" in analysis:  
            return (  
                result,  
                f"{conf_val:.3f}",  
                str(analysis),  
                None,  
                None,  
                None,  
                None,  
                "<div style='padding:10px; background-color:#f8d7da; border:1px solid #f5c6cb; border-radius:5px; color:#721c24;'>Error occurred: Empty sequence provided.</div>",  
            )  
  
        # save amino acid composition plot  
        output_file = None  
        comp_plot = None  
        if plot_buf:  
            comp_plot = plot_buf.getvalue()  
            unique_name = f"composition_{uuid.uuid4().hex}.png"  
            with open(unique_name, "wb") as f:  
                f.write(comp_plot)  
            output_file = unique_name  
            schedule_delete(output_file)  
  
        # probability bar chart  
        prob_chart = make_prob_chart(conf_val)  
  
        # save report files  
        csv_file = save_csv(analysis, f"report_{uuid.uuid4().hex}.csv")  
        pdf_file = save_pdf(analysis, f"report_{uuid.uuid4().hex}.pdf")  
  
        # warning  
        warning_html = "<div style='padding:10px; background-color:#fff3cd; border:1px solid #ffeeba; border-radius:5px; color:#856404;'>Temporary files (plots/reports) will be auto-deleted after 15 minutes.</div>"  
  
        return (  
            result,  
            f"{conf_val:.3f}",  
            str(analysis),  
            comp_plot,  
            prob_chart,  
            csv_file,  
            pdf_file,  
            warning_html,  
        )  
  
    except Exception as e:  
        return (  
            "Error",  
            "0.000",  
            str(e),  
            None,  
            None,  
            None,  
            None,  
            f"<div style='padding:10px; background-color:#f8d7da; border:1px solid #f5c6cb; border-radius:5px; color:#721c24;'>Error occurred: {e}</div>",  
        )  
  
# --- Gradio interface ---  
with gr.Blocks(theme=gr.themes.Soft()) as demo:  
    gr.Markdown("## Protein Enzyme Classifier (with Advanced Analysis)")  
    seq_input = gr.Textbox(lines=5, label="Protein Sequence", placeholder="Enter protein sequence...")  
    model_choice = gr.Dropdown(["CNN", "LSTM"], value="CNN", label="Choose Model")  
  
    with gr.Row():  
        result_out = gr.Label(label="Prediction")  
        conf_out = gr.Label(label="Confidence")  
  
    analysis_out = gr.Textbox(label="Detailed Analysis")  
    with gr.Row():  
        plot_out = gr.Image(label="Amino Acid Composition")  
        prob_out = gr.Image(label="Probability Chart")  
  
    with gr.Row():  
        file_csv = gr.File(label="Download CSV Report")  
        file_pdf = gr.File(label="Download PDF Report")  
  
    warning_out = gr.HTML()  
  
    run_btn = gr.Button("Run Analysis")  
    run_btn.click(  
        fn=predict_protein,  
        inputs=[seq_input, model_choice],  
        outputs=[result_out, conf_out, analysis_out, plot_out, prob_out, file_csv, file_pdf, warning_out],  
    )  
  
if __name__ == "__main__":  
    demo.launch()
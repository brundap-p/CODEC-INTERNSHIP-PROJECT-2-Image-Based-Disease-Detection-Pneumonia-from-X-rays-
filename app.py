import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="PneumoXAI Dashboard",
    page_icon="🫁",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

metrics_df = pd.read_csv("data/project_metrics.csv")
history_df = pd.read_csv("data/training_history.csv")
comparison_df = pd.read_csv("data/model_comparison.csv")
pixel_df = pd.read_csv("data/pixel_statistics.csv")

metrics = dict(
    zip(
        metrics_df["Metric"],
        metrics_df["Value"]
    )
)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.image(
    "images/lungs.png",
    use_container_width=True
)

st.sidebar.title("🫁 PneumoXAI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Analysis",
        "Model Architecture",
        "Training Performance",
        "Model Evaluation",
        "Explainable AI",
        "About Project"
    ]
)

# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.title("🫁 PneumoXAI Dashboard")

    st.subheader(
        "Explainable Pneumonia Detection Using DenseNet121 and Grad-CAM"
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['Accuracy']}%"
        )

    with col2:
        st.metric(
            "ROC-AUC",
            round(metrics["ROC_AUC"], 4)
        )

    with col3:
        st.metric(
            "Train Images",
            int(metrics["Train_Images"])
        )

    with col4:
        st.metric(
            "Test Images",
            int(metrics["Test_Images"])
        )

    st.markdown("---")

    st.image(
        "images/lungs3.png",
        use_container_width=True
    )

    st.markdown("---")

    st.info(
        """
        PneumoXAI is an Explainable Artificial Intelligence (XAI)
        system designed for automatic pneumonia detection from
        chest X-ray images.

        The project utilizes DenseNet121 for image classification
        and Grad-CAM for visual explanations of model decisions.
        """
    )

# ==================================================
# DATASET ANALYSIS
# ==================================================

elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    dataset_df = pd.DataFrame(
        {
            "Class": [
                "NORMAL",
                "PNEUMONIA"
            ],
            "Count": [
                int(metrics["Normal_Count"]),
                int(metrics["Pneumonia_Count"])
            ]
        }
    )

    col1, col2 = st.columns(2)

    with col1:

        pie_fig = px.pie(
            dataset_df,
            names="Class",
            values="Count",
            title="Balanced Dataset Distribution"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

    with col2:

        bar_fig = px.bar(
            dataset_df,
            x="Class",
            y="Count",
            color="Class",
            text="Count",
            title="Class-wise Image Count"
        )

        st.plotly_chart(
            bar_fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Sample Chest X-ray Images")

    st.image(
        "images/sample_xrays.png",
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Normal vs Pneumonia Comparison")

    st.image(
        "images/normal_vs_pneumonia.png",
        use_container_width=True
    )

# ==================================================
# MODEL ARCHITECTURE
# ==================================================

elif page == "Model Architecture":

    st.title("🧠 Model Architecture")

    st.markdown(
        """
        ### DenseNet121 Architecture

        DenseNet121 was selected as the final model
        because it achieved the highest test accuracy
        among all evaluated architectures.

        Pipeline:

        Input Image (224×224)

        ↓

        DenseNet121 Backbone

        ↓

        Global Average Pooling

        ↓

        Dense Layer (256)

        ↓

        Output Layer (NORMAL / PNEUMONIA)
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            DenseNet121 employs dense connectivity,
            where each layer receives feature maps
            from all preceding layers.

            Advantages:

            • Improved feature propagation

            • Reduced vanishing gradient problem

            • Better parameter efficiency

            • Strong performance on medical imaging tasks
            """
        )

    with col2:

        st.success(
            """
            Final Model Configuration

            Input Shape : 224 × 224 × 3

            Backbone : DenseNet121

            Pooling : Global Average Pooling

            Dense Layer : 256 Units

            Output Classes : 2

            Classification :
            NORMAL / PNEUMONIA
            """
        )

    st.markdown("---")

    st.subheader("Model Accuracy Comparison")

    comparison_fig = px.bar(
        comparison_df,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy",
        title="Model Performance Comparison"
    )

    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )

    st.success(
        "DenseNet121 achieved the highest test accuracy (83.73%) and was selected as the final model."
    )

# ==================================================
# TRAINING PERFORMANCE
# ==================================================

elif page == "Training Performance":

    st.title("📈 Training Performance")

    col1, col2 = st.columns(2)

    with col1:

        acc_fig = px.line(
            history_df,
            x="Epoch",
            y="Accuracy",
            markers=True,
            title="Training Accuracy"
        )

        st.plotly_chart(
            acc_fig,
            use_container_width=True
        )

    with col2:

        loss_fig = px.line(
            history_df,
            x="Epoch",
            y="Loss",
            markers=True,
            title="Training Loss"
        )

        st.plotly_chart(
            loss_fig,
            use_container_width=True
        )

    st.success(
        "DenseNet121 achieved stable convergence with increasing accuracy and decreasing loss."
    )

# ==================================================
# MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":

    st.title("📋 Model Evaluation")

    st.subheader("Confusion Matrix")

    cm = pd.read_csv(
        "data/confusion_matrix.csv",
        index_col=0
    )

    heatmap_fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix"
    )

    st.plotly_chart(
        heatmap_fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "Precision / Recall / F1 Score"
    )

    metrics_chart = pd.DataFrame(
        {
            "Class": [
                "NORMAL",
                "NORMAL",
                "NORMAL",
                "PNEUMONIA",
                "PNEUMONIA",
                "PNEUMONIA"
            ],

            "Metric": [
                "Precision",
                "Recall",
                "F1 Score",
                "Precision",
                "Recall",
                "F1 Score"
            ],

            "Value": [
                0.99,
                0.76,
                0.86,
                0.68,
                0.98,
                0.81
            ]
        }
    )

    metric_fig = px.bar(
        metrics_chart,
        x="Class",
        y="Value",
        color="Metric",
        barmode="group",
        title="Performance Metrics"
    )

    st.plotly_chart(
        metric_fig,
        use_container_width=True
    )

    st.metric(
        "ROC-AUC Score",
        "0.9689",
        "A ROC-AUC value close to 1 indicates excellent discrimination between NORMAL and PNEUMONIA classes."
    )

# ==================================================
# EXPLAINABLE AI
# ==================================================

elif page == "Explainable AI":

    st.title("🔍 Explainable AI")

    st.info(
        """
        Grad-CAM is used to visualize
        regions that most influenced
        the model's prediction.
        """
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Grad-CAM Heatmap",
            "Overlay Visualization",
            "Highest Model Attention Region"
        ]
    )

    with tab1:

        st.image(
            "images/gradcam_heatmap.png",
            use_container_width=True
        )

    with tab2:

        st.image(
            "images/overlay.png",
            use_container_width=True
        )

    with tab3:

        st.image(
            "images/roi.png",
            use_container_width=True
        )

    st.markdown("---")

    st.subheader(
        "Pixel Statistics Heatmap"
    )

    pixel_fig = px.imshow(
        pixel_df.set_index("Metric").T,
        text_auto=True,
        aspect="auto",
        title="Pixel Statistics Analysis"
    )

    st.plotly_chart(
        pixel_fig,
        use_container_width=True
    )

    st.success(
        """
        The highlighted region represents
        the Highest Model Attention Region.

        It indicates the area that contributed
        most strongly to the prediction
        according to Grad-CAM.
        """
    )

# ==================================================
# ABOUT PROJECT
# ==================================================

elif page == "About Project":

    st.title("ℹ️ About Project")

    st.markdown(
        """
        ### PneumoXAI

        Explainable Pneumonia Detection
        using Deep Learning and Grad-CAM.

        ### Dataset

        NORMAL Images : 7758

        PNEUMONIA Images : 7758

        Balanced Dataset Size : 15516

        ### Technologies Used

        • Python

        • TensorFlow

        • Keras

        • DenseNet121

        • Grad-CAM

        • Gradio

        • Streamlit

        • Plotly

        ### Final Results

        Accuracy : 83.73%

        ROC-AUC : 0.9689

        Best Model : DenseNet121
        """
    )

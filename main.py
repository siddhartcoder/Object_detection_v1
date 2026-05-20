"""
Main CLI Entry Point — BDD100K Object Detection Assignment.

Supports five tasks:
    analysis  — Run full EDA on BDD100K annotations
    dashboard — Launch interactive Plotly Dash dashboard
    model     — Run YOLOv8 inference on validation images
    train     — Fine-tune YOLOv8 on BDD100K subset (1 epoch demo)
    evaluate  — Evaluate model and produce visualizations

Usage::

    python main.py --task analysis \\
        --train_json data/labels/det_20/det_train.json \\
        --val_json data/labels/det_20/det_val.json \\
        --image_dir data/images/100k

    python main.py --task dashboard \\
        --train_json data/labels/det_20/det_train.json \\
        --val_json data/labels/det_20/det_val.json

    python main.py --task model \\
        --val_json data/labels/det_20/det_val.json \\
        --image_dir data/images/100k/val \\
        --output_dir results/predictions/

    python main.py --task train \\
        --train_json data/labels/det_20/det_train.json \\
        --image_dir data/images/100k/train \\
        --subset_size 1000 --epochs 1

    python main.py --task evaluate \\
        --val_json data/labels/det_20/det_val.json \\
        --image_dir data/images/100k/val \\
        --output_dir results/evaluation/
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed Namespace object with all arguments.
    """
    parser = argparse.ArgumentParser(
        description="BDD100K Object Detection — Applied CV Assignment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["analysis", "dashboard", "model", "train", "evaluate"],
        help=(
            "Task to run:\n"
            "  analysis  — Full EDA and plots\n"
            "  dashboard — Interactive Plotly Dash app\n"
            "  model     — YOLOv8 inference on val images\n"
            "  train     — Fine-tune YOLOv8 on subset\n"
            "  evaluate  — Evaluate model + visualizations"
        ),
    )
    parser.add_argument("--train_json", default="Data/labels/bdd100k_labels_images_train.json",
                        help="Path to BDD100K training annotation JSON.")
    parser.add_argument("--val_json", default="Data/labels/bdd100k_labels_images_val.json",
                        help="Path to BDD100K validation annotation JSON.")
    parser.add_argument("--image_dir", default="Data/images",
                        help="Root image directory (contains train/ and val/ subfolders).")
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save outputs.")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 model name or path to .pt checkpoint.")
    parser.add_argument("--subset_size", type=int, default=1000,
                        help="Number of training images for subset training (default: 1000).")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 1).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size (default: 8).")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Image size for training/inference (default: 640).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detections (default: 0.25).")
    parser.add_argument("--device", default="cpu",
                        help="Device for inference/training: cpu, cuda, cuda:0 (default: cpu).")
    parser.add_argument("--max_eval_images", type=int, default=None,
                        help="Max images to evaluate (default: all).")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port for dashboard (default: 8050).")
    return parser.parse_args()


def run_analysis(args: argparse.Namespace) -> None:
    """Execute the EDA analysis task.

    Args:
        args: Parsed CLI arguments.
    """
    from src.data_analysis.bdd_parser import BDDDataset
    from src.data_analysis.analysis import run_full_analysis

    dataset = BDDDataset(
        train_json=args.train_json,
        val_json=args.val_json,
        image_dir=args.image_dir,
    )
    dataset.load()
    output_dir = os.path.join(args.output_dir, "analysis")
    run_full_analysis(dataset, output_dir=output_dir)


def run_dashboard(args: argparse.Namespace) -> None:
    """Launch the interactive Plotly Dash dashboard.

    Args:
        args: Parsed CLI arguments.
    """
    from src.data_analysis.bdd_parser import BDDDataset
    from src.data_analysis.dashboard import build_dashboard

    dataset = BDDDataset(
        train_json=args.train_json,
        val_json=args.val_json,
    )
    dataset.load()
    app = build_dashboard(dataset)
    print(f"\n[Dashboard] Starting at http://localhost:{args.port}")
    print("[Dashboard] Press Ctrl+C to stop.\n")
    app.run(debug=False, host="0.0.0.0", port=args.port)


def run_model(args: argparse.Namespace) -> None:
    """Run YOLOv8 inference on validation images.

    Args:
        args: Parsed CLI arguments.
    """
    import json
    from src.data_analysis.bdd_parser import BDDDataset
    from src.model.model_loader import YOLOv8Detector

    val_image_dir = os.path.join(args.image_dir, "val") if not args.image_dir.endswith("val") else args.image_dir

    # Collect image paths
    image_paths = [
        os.path.join(val_image_dir, f)
        for f in os.listdir(val_image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ] if os.path.exists(val_image_dir) else []

    if not image_paths:
        print(f"[model] No images found in: {val_image_dir}")
        sys.exit(1)

    print(f"[model] Found {len(image_paths)} images in: {val_image_dir}")

    detector = YOLOv8Detector(
        model_name=args.model,
        conf_threshold=args.conf,
        device=args.device,
    )
    detector.load()

    all_detections = detector.predict_batch(image_paths, batch_size=16)

    # Save predictions to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    results = [
        det.to_dict()
        for dets in all_detections.values()
        for det in dets
    ]
    out_path = os.path.join(args.output_dir, "predictions.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[model] Saved {len(results)} detections to: {out_path}")


def run_train(args: argparse.Namespace) -> None:
    """Fine-tune YOLOv8 on a BDD100K subset.

    Args:
        args: Parsed CLI arguments.
    """
    val_image_dir = os.path.join(args.image_dir.replace("train", ""), "val")
    train_image_dir = args.image_dir if "train" in args.image_dir else os.path.join(args.image_dir, "train")

    from src.model.trainer import BDDYOLOTrainer

    trainer = BDDYOLOTrainer(
        train_json=args.train_json,
        val_json=args.val_json,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        output_dir=os.path.join(args.output_dir, "training"),
        model_name=args.model,
        subset_size=args.subset_size,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    trainer.prepare_dataset()
    trainer.train()


def run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate YOLOv8 on BDD100K val and produce all visualizations.

    Args:
        args: Parsed CLI arguments.
    """
    from src.data_analysis.bdd_parser import BDDDataset
    from src.model.model_loader import YOLOv8Detector
    from src.evaluation.evaluator import BDDEvaluator
    from src.evaluation.visualizer import (
        plot_map_barchart,
        plot_confusion_matrix,
        visualize_predictions,
        plot_failure_gallery,
        plot_failure_clusters,
        plot_missed_object_sizes,
    )

    val_image_dir = args.image_dir if args.image_dir.endswith("val") else os.path.join(args.image_dir, "val")

    # Load dataset
    dataset = BDDDataset(train_json=args.train_json, val_json=args.val_json, image_dir=args.image_dir)
    dataset.load()

    # Load model
    detector = YOLOv8Detector(
        model_name=args.model,
        conf_threshold=args.conf,
        device=args.device,
    )
    detector.load()

    # Run evaluation
    evaluator = BDDEvaluator(
        detector=detector,
        val_frames=dataset.val_frames,
        val_image_dir=val_image_dir,
    )
    results = evaluator.evaluate(max_images=args.max_eval_images, output_dir=args.output_dir)
    evaluator.print_report(results)
    evaluator.save_results(results, args.output_dir)

    # Quantitative visualizations
    quant_dir = os.path.join(args.output_dir, "quantitative")
    plot_map_barchart(results["per_class_ap"], results["mAP_50"], quant_dir)
    plot_confusion_matrix(results["confusion_matrix"], quant_dir)

    # Qualitative visualizations — need full predictions
    print("\n[evaluate] Running full inference for qualitative visualizations ...")
    val_frames = dataset.val_frames
    if args.max_eval_images:
        val_frames = val_frames[: args.max_eval_images]

    image_paths = [os.path.join(val_image_dir, f.name) for f in val_frames if os.path.exists(os.path.join(val_image_dir, f.name))]
    all_detections = detector.predict_batch(image_paths, batch_size=16)

    qual_dir = os.path.join(args.output_dir, "qualitative")
    visualize_predictions(val_frames, all_detections, val_image_dir, os.path.join(qual_dir, "overlays"), n_samples=20)
    failure_df = plot_failure_gallery(val_frames, all_detections, val_image_dir, os.path.join(qual_dir, "failures"))
    plot_failure_clusters(failure_df, qual_dir)
    plot_missed_object_sizes(val_frames, all_detections, qual_dir)

    print(f"\n[evaluate] All outputs saved to: {args.output_dir}")


TASK_RUNNERS = {
    "analysis": run_analysis,
    "dashboard": run_dashboard,
    "model": run_model,
    "train": run_train,
    "evaluate": run_evaluate,
}


if __name__ == "__main__":
    args = parse_args()
    runner = TASK_RUNNERS.get(args.task)
    if runner is None:
        print(f"Unknown task: {args.task}")
        sys.exit(1)
    runner(args)

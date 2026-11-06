"""Pick the most accurate people counter that can run on this machine."""


def make_counter():
    """The most accurate counter that can run here: point model, density model, then YOLO."""
    from crowdshield.monitor import points_to_grid

    try:
        from crowdshield.points import load_point_model, point_grid

        load_point_model()
        return point_grid, "point model (PET)"
    except Exception:
        pass
    try:
        from crowdshield.density import density_grid, load_density_model, DEFAULT_WEIGHTS

        load_density_model(str(DEFAULT_WEIGHTS))
        return density_grid, "density model (DM-Count)"
    except Exception:
        from crowdshield.detection import detect_people, load_detector

        if load_detector("crowd_yolo") is None:
            return None, "no counting model is installed (needs PyTorch and Ultralytics)"

        def yolo_grid(frame):
            boxes = detect_people(frame)["person_boxes"]
            centres = [(y + h / 2, x + w / 2) for x, y, w, h in boxes]
            return points_to_grid(centres, frame.shape)

        return yolo_grid, "YOLO (density model unavailable)"

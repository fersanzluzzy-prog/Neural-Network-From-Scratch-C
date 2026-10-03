from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples" / "output"
BUILD = ROOT / "build"
DEMO_EXE = BUILD / "demo_training.exe"


def read_dataset(path: Path):
    xs = []
    ys = []
    labels = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            labels.append(int(float(row["label"])))

    return xs, ys, labels


def read_training(path: Path):
    snapshots = {}
    losses = {}

    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            line = raw.strip()

            if not line:
                continue

            if line.startswith("# EPOCH"):
                parts = line.split()

                epoch = int(parts[2])
                loss = float(parts[4])

                snapshots[epoch] = []
                losses[epoch] = loss
                continue

            if line.startswith("x,y"):
                continue

            x, y, prediction, epoch = line.split(",")
            epoch = int(epoch)

            if epoch not in snapshots:
                snapshots[epoch] = []

            snapshots[epoch].append(
                (float(x), float(y), float(prediction))
            )

    epochs = sorted(snapshots)

    return epochs, snapshots, losses


class TrainingViewer:
    def __init__(self, name: str):
        self.name = name

        self.data_path = OUTPUT / f"{name}_data.csv"
        self.training_path = OUTPUT / f"{name}_training.csv"

        if not self.data_path.exists() or not self.training_path.exists():
            raise FileNotFoundError(
                "No existen los datos. Ejecuta primero "
                "demo_training.exe."
            )

        self.xs, self.ys, self.labels = read_dataset(self.data_path)
        self.epochs, self.snapshots, self.losses = read_training(
            self.training_path
        )

        if not self.epochs:
            raise RuntimeError("No hay snapshots de entrenamiento.")

        self.class0 = [
            i for i, label in enumerate(self.labels) if label == 0
        ]
        self.class1 = [
            i for i, label in enumerate(self.labels) if label == 1
        ]

        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Dejamos espacio inferior para el botón.
        self.fig.subplots_adjust(bottom=0.13)

        button_ax = self.fig.add_axes([0.35, 0.025, 0.30, 0.055])

        self.reset_button = Button(
            button_ax,
            "Resetear entrenamiento"
        )
        self.reset_button.on_clicked(self.reset)

        self.animation = None
        self.start_animation()

    def draw_frame(self, frame: int):
        epoch = self.epochs[frame]
        values = self.snapshots[epoch]

        grid_x = [value[0] for value in values]
        grid_y = [value[1] for value in values]
        predictions = [value[2] for value in values]

        self.ax.clear()

        self.ax.tricontourf(
            grid_x,
            grid_y,
            predictions,
            levels=[0.0, 0.5, 1.0]
        )

        self.ax.scatter(
            [self.xs[i] for i in self.class0],
            [self.ys[i] for i in self.class0],
            marker="o",
            s=45,
            label="Clase 0"
        )

        self.ax.scatter(
            [self.xs[i] for i in self.class1],
            [self.ys[i] for i in self.class1],
            marker="x",
            s=55,
            label="Clase 1"
        )

        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_aspect("equal", adjustable="box")

        self.ax.set_xlabel("Entrada 1")
        self.ax.set_ylabel("Entrada 2")

        self.ax.set_title(
            f"Decision Boundary — {self.name.upper()}\n"
            f"Época: {epoch}   Loss: {self.losses[epoch]:.6f}"
        )

        self.ax.legend()

        self.fig.canvas.draw_idle()

    def start_animation(self):
        if self.animation is not None:
            self.animation.event_source.stop()

        self.animation = FuncAnimation(
            self.fig,
            self.draw_frame,
            frames=len(self.epochs),
            interval=120,
            repeat=False
        )

        # Mantener referencia.
        self.fig._training_animation = self.animation

    def reset(self, _event):
        """
        Vuelve a ejecutar demo_training.exe desde cero y recarga
        los CSV generados por C. Así el botón reinicia realmente
        el entrenamiento, no solamente la animación.
        """
        if not DEMO_EXE.exists():
            print(f"No se encuentra: {DEMO_EXE}")
            return

        if self.animation is not None:
            self.animation.event_source.stop()

        print("Reiniciando entrenamiento...")

        result = subprocess.run(
            [str(DEMO_EXE)],
            cwd=str(ROOT),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("Error ejecutando demo_training.exe:")
            print(result.stdout)
            print(result.stderr)
            return

        self.xs, self.ys, self.labels = read_dataset(self.data_path)
        self.epochs, self.snapshots, self.losses = read_training(
            self.training_path
        )

        self.class0 = [
            i for i, label in enumerate(self.labels) if label == 0
        ]
        self.class1 = [
            i for i, label in enumerate(self.labels) if label == 1
        ]

        self.start_animation()
        self.fig.canvas.draw_idle()


def main():
    if len(sys.argv) != 2:
        print("Uso:")
        print("  python examples/plot_training.py xor")
        print("  python examples/plot_training.py circles")
        return 1

    name = sys.argv[1].lower()

    if name not in {"xor", "circles"}:
        print("Usa 'xor' o 'circles'.")
        return 1

    TrainingViewer(name)
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

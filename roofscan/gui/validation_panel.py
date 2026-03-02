"""Panel de validación y feedback para la GUI de RoofScan.

Permite al usuario:
1. Ingresar el área medida manualmente y comparar con la predicción (MAPE).
2. Guardar la detección actual como par de entrenamiento (feedback).
3. Lanzar el reentrenamiento del modelo U-Net con los datos acumulados.
4. Ver estadísticas del historial de validaciones y del dataset de feedback.

Se conecta a la ventana principal a través de señales Qt. No accede
directamente a archivos de imagen ni modelos — recibe esos datos desde
``MainWindow`` cuando el usuario completa un análisis.
"""

import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


class ValidationPanel(QWidget):
    """Panel lateral de validación y feedback.

    Atributos internos de estado:
        _current_pred_area_m2: Área predicha en la última detección.
        _current_array: Array de imagen del último análisis.
        _current_mask: Máscara del último análisis.
        _current_label: Etiqueta descriptiva (ej. dirección) del análisis.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pred_area_m2: float | None = None
        self._current_array: np.ndarray | None = None
        self._current_mask: np.ndarray | None = None
        self._current_label: str = ""

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        layout.addWidget(self._build_area_group())
        layout.addWidget(self._build_feedback_group())
        layout.addWidget(self._build_history_group())
        layout.addStretch()

    # ------------------------------------------------------------------
    # API pública — llamada desde MainWindow
    # ------------------------------------------------------------------

    def set_analysis_result(
        self,
        pred_area_m2: float,
        image_array: np.ndarray,
        mask: np.ndarray,
        label: str = "",
    ) -> None:
        """Actualiza el panel con los resultados del último análisis.

        Args:
            pred_area_m2: Área total predicha en m².
            image_array: Array normalizado ``(bandas, H, W)``.
            mask: Máscara booleana ``(H, W)``.
            label: Etiqueta descriptiva del análisis (ej. dirección).
        """
        self._current_pred_area_m2 = pred_area_m2
        self._current_array = image_array
        self._current_mask = mask
        self._current_label = label

        self._lbl_pred_area.setText(f"{pred_area_m2:,.1f} m²")
        self._lbl_mape.setText("—")
        self._lbl_mape.setStyleSheet("")
        self._btn_calc.setEnabled(True)
        self._btn_save_feedback.setEnabled(True)
        self._refresh_feedback_stats()

    def update_retrain_state(self, running: bool) -> None:
        """Actualiza el botón de reentrenamiento según si está en curso."""
        self._btn_retrain.setEnabled(not running)
        self._btn_retrain.setText(
            "Reentrenando..." if running else "Reentrenar modelo U-Net"
        )

    def show_retrain_result(self, history: dict) -> None:
        """Muestra el resultado del reentrenamiento."""
        best_epoch = history.get("best_epoch", "?")
        best_val = history.get("best_val_loss", float("nan"))
        self._lbl_retrain_result.setText(
            f"Listo — mejor época: {best_epoch} | val_loss: {best_val:.4f}"
        )
        self._btn_retrain.setEnabled(True)
        self._btn_retrain.setText("Reentrenar modelo U-Net")
        self._refresh_feedback_stats()

    def show_retrain_error(self, msg: str) -> None:
        """Muestra un mensaje de error del reentrenamiento."""
        self._lbl_retrain_result.setText(f"Error: {msg}")
        self._btn_retrain.setEnabled(True)
        self._btn_retrain.setText("Reentrenar modelo U-Net")

    # ------------------------------------------------------------------
    # Slots internos
    # ------------------------------------------------------------------

    def _on_calc_mape(self) -> None:
        """Calcula y muestra el MAPE cuando el usuario presiona el botón."""
        if self._current_pred_area_m2 is None:
            return

        manual = self._spin_manual.value()
        if manual <= 0:
            self._lbl_mape.setText("Ingresá un área > 0")
            return

        from roofscan.config import FEEDBACK_DIR
        from roofscan.core.validacion.metrics import log_validation_result, validate

        result = validate(
            pred_area_m2=self._current_pred_area_m2,
            manual_area_m2=manual,
            label=self._current_label,
        )

        log_validation_result(result, FEEDBACK_DIR / "validation_log.csv")

        mape = result.mape_pct
        color = "#28a745" if mape <= 10 else ("#ffc107" if mape <= 20 else "#dc3545")
        self._lbl_mape.setText(f"{mape:.1f}%")
        self._lbl_mape.setStyleSheet(f"color: {color}; font-weight: bold;")

        # Actualizar historial en el panel
        self._refresh_history_label()
        log.info("MAPE calculado: %.1f%% (pred=%.1f, manual=%.1f)", mape,
                 self._current_pred_area_m2, manual)

    def _on_save_feedback(self) -> None:
        """Guarda el par (imagen, máscara) actual en el dataset de feedback."""
        if self._current_array is None or self._current_mask is None:
            return

        from roofscan.config import FEEDBACK_DIR
        from roofscan.core.validacion.feedback_store import save_feedback_pair

        try:
            name = save_feedback_pair(
                self._current_array,
                self._current_mask,
                FEEDBACK_DIR,
            )
            self._refresh_feedback_stats()
            log.info("Feedback guardado como: %s", name)
        except Exception as exc:
            log.error("Error guardando feedback: %s", exc)
            self._lbl_feedback_count.setText(f"Error: {exc}")

    def _on_request_retrain(self) -> None:
        """Emite la solicitud de reentrenamiento hacia la ventana principal."""
        # Usamos el mecanismo de señal a través del parent (MainWindow)
        mw = self.window()
        if hasattr(mw, "_start_retrain"):
            mw._start_retrain()
        else:
            log.warning("MainWindow sin método _start_retrain; reentrenamiento ignorado.")

    # ------------------------------------------------------------------
    # Builders de sub-widgets
    # ------------------------------------------------------------------

    def _build_area_group(self) -> QGroupBox:
        grp = QGroupBox("Error de área")
        grp.setStyleSheet("QGroupBox { font-weight: bold; }")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # Área predicha
        row_pred = QHBoxLayout()
        row_pred.addWidget(QLabel("Área automática:"))
        self._lbl_pred_area = QLabel("—")
        self._lbl_pred_area.setAlignment(Qt.AlignmentFlag.AlignRight)
        row_pred.addWidget(self._lbl_pred_area)
        lay.addLayout(row_pred)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        lay.addWidget(sep)

        # Área manual
        row_manual = QHBoxLayout()
        row_manual.addWidget(QLabel("Área manual (m²):"))
        self._spin_manual = QDoubleSpinBox()
        self._spin_manual.setRange(0.0, 1_000_000.0)
        self._spin_manual.setDecimals(1)
        self._spin_manual.setSingleStep(10.0)
        self._spin_manual.setFixedWidth(110)
        self._spin_manual.setAlignment(Qt.AlignmentFlag.AlignRight)
        row_manual.addWidget(self._spin_manual)
        lay.addLayout(row_manual)

        # Botón calcular
        self._btn_calc = QPushButton("Calcular error")
        self._btn_calc.setEnabled(False)
        self._btn_calc.clicked.connect(self._on_calc_mape)
        lay.addWidget(self._btn_calc)

        # Resultado MAPE
        row_mape = QHBoxLayout()
        row_mape.addWidget(QLabel("Error (MAPE):"))
        self._lbl_mape = QLabel("—")
        self._lbl_mape.setAlignment(Qt.AlignmentFlag.AlignRight)
        row_mape.addWidget(self._lbl_mape)
        lay.addLayout(row_mape)

        return grp

    def _build_feedback_group(self) -> QGroupBox:
        grp = QGroupBox("Feedback para reentrenamiento")
        grp.setStyleSheet("QGroupBox { font-weight: bold; }")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # Contador de pares
        row_count = QHBoxLayout()
        row_count.addWidget(QLabel("Pares acumulados:"))
        self._lbl_feedback_count = QLabel("0")
        self._lbl_feedback_count.setAlignment(Qt.AlignmentFlag.AlignRight)
        row_count.addWidget(self._lbl_feedback_count)
        lay.addLayout(row_count)

        # Botón guardar detección
        self._btn_save_feedback = QPushButton("Guardar detección actual")
        self._btn_save_feedback.setEnabled(False)
        self._btn_save_feedback.setToolTip(
            "Guarda la imagen y máscara del análisis actual\n"
            "como dato de entrenamiento para el modelo U-Net."
        )
        self._btn_save_feedback.clicked.connect(self._on_save_feedback)
        lay.addWidget(self._btn_save_feedback)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        lay.addWidget(sep)

        # Botón reentrenar
        self._btn_retrain = QPushButton("Reentrenar modelo U-Net")
        self._btn_retrain.setToolTip(
            "Inicia el fine-tuning del modelo con todos los pares\n"
            "de feedback acumulados. Puede tardar varios minutos."
        )
        self._btn_retrain.clicked.connect(self._on_request_retrain)
        lay.addWidget(self._btn_retrain)

        # Resultado del reentrenamiento
        self._lbl_retrain_result = QLabel("")
        self._lbl_retrain_result.setWordWrap(True)
        self._lbl_retrain_result.setStyleSheet("color: #6c757d; font-size: 10px;")
        lay.addWidget(self._lbl_retrain_result)

        return grp

    def _build_history_group(self) -> QGroupBox:
        grp = QGroupBox("Historial de validaciones")
        grp.setStyleSheet("QGroupBox { font-weight: bold; }")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        self._lbl_history = QLabel("Sin validaciones registradas.")
        self._lbl_history.setWordWrap(True)
        self._lbl_history.setStyleSheet("color: #6c757d; font-size: 10px;")
        lay.addWidget(self._lbl_history)

        btn_refresh = QPushButton("Actualizar estadísticas")
        btn_refresh.clicked.connect(self._refresh_history_label)
        lay.addWidget(btn_refresh)

        return grp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_feedback_stats(self) -> None:
        """Actualiza el contador de pares de feedback."""
        try:
            from roofscan.config import FEEDBACK_DIR
            from roofscan.core.validacion.feedback_store import count_feedback_pairs
            n = count_feedback_pairs(FEEDBACK_DIR)
            self._lbl_feedback_count.setText(str(n))
            self._btn_retrain.setEnabled(n >= 2)
            if n < 2:
                self._lbl_retrain_result.setText(
                    "Se necesitan al menos 2 pares para reentrenar."
                )
        except Exception as exc:
            log.debug("No se pudo actualizar stats de feedback: %s", exc)

    def _refresh_history_label(self) -> None:
        """Actualiza el resumen del historial de validaciones."""
        try:
            from roofscan.config import FEEDBACK_DIR
            from roofscan.core.validacion.metrics import summary_stats
            stats = summary_stats(FEEDBACK_DIR / "validation_log.csv")
            n = stats["n_validations"]
            if n == 0:
                self._lbl_history.setText("Sin validaciones registradas.")
                return
            lines = [f"Total: {n} validación(es)"]
            if stats["mean_mape_pct"] is not None:
                lines.append(f"MAPE promedio: {stats['mean_mape_pct']:.1f}%")
            if stats["mean_iou"] is not None:
                lines.append(f"IoU promedio: {stats['mean_iou']:.3f}")
            if stats["mean_f1"] is not None:
                lines.append(f"F1 promedio: {stats['mean_f1']:.3f}")
            self._lbl_history.setText("\n".join(lines))
        except Exception as exc:
            log.debug("No se pudo leer historial: %s", exc)

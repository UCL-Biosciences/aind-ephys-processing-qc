from __future__ import annotations

from pathlib import Path
import logging
import json
from datetime import datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import welch, savgol_filter

import spikeinterface as si
from spikeinterface.core.core_tools import check_json
import spikeinterface.preprocessing as spre
from spikeinterface.curation import validate_curation_dict
import spikeinterface.widgets as sw

from aind_data_schema_models.modalities import Modality

from aind_data_schema.core.processing import Processing
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status, Stage, CurationMetric, CurationHistory
from aind_qcportal_schema.metric_value import DropdownMetric


default_curation_dict = {
    "format_version": "2",
    "label_definitions": {
        "quality":{
            "label_options": ["good", "MUA", "noise"],
            "exclusive": True,
        }, 
    },
    "manual_labels": [],
    "removed": [],
    "merges": [],
    "splits": [],
}


def _get_fig_axs(nrows, ncols, subplot_figsize=(3, 3)):
    figsize = (subplot_figsize[0] * ncols, subplot_figsize[1] * nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])[:, None]
    elif nrows == 1:
        axs = axs[None, :]
    elif ncols == 1:
        axs = axs[:, None]
    return fig, axs

def _get_surface_channel(recording: si.BaseRecording, channel_labels: np.ndarray) -> int | None:
    y_locs = recording.get_channel_locations()[:, 1]
    surface_channel_y_position = None
    if channel_labels is not None:
        channel_ids_out = recording.channel_ids[channel_labels == 'out']

        if len(channel_ids_out) == 0:
           logging.info("No out channels detected")
        else:
            surface_channel_id = channel_ids_out[0]
            surface_channel_index = recording.channel_ids.tolist().index(surface_channel_id)
            surface_channel_y_position = y_locs[surface_channel_index]
    
    return surface_channel_y_position

def recording_abbrv_name(recording_name):
    """
    Abbreviates recording name from Open Ephys stream by removing Record Node and Plugin source from the name.
    """
    import re
    if "Record Node" in recording_name:
        return re.sub(r'Record Node [^#]+#Neuropix-PXI-\d+\.', '', recording_name)
    else:
        return recording_name


def load_preprocessed_recording(preprocessed_json_file, session_name, ecephys_folder, data_folder):
    """
    Loads preprocessed recording from preprocessed JSON file.
    """
    recording_preprocessed = None
    if preprocessed_json_file.is_file():
        try:
            recording_preprocessed = si.load(preprocessed_json_file, base_folder=data_folder)
        except Exception as e:
            pass
        if recording_preprocessed is None:
            try:
                json_str = json.dumps(json.load(open(preprocessed_json_file)))
                if "ecephys_session" in json_str:
                    json_str_remapped = json_str.replace("ecephys_session", ecephys_folder.name)
                elif ecephys_folder.name != session_name and session_name in json_str:
                    json_str_remapped = json_str.replace(session_name, ecephys_folder.name)
                else:
                    json_str_remapped = json_str
                recording_dict = json.loads(json_str_remapped)
                recording_preprocessed = si.load(recording_dict, base_folder=data_folder)
            except:
                pass
        if recording_preprocessed is None:
            # UCL patch (why we're here): the saved recording's folder_path is
            # a relative path like "../../../../data/spikeglx/concat_session".
            # It was correct at save time, but Nextflow tasks don't all sit at
            # the same folder depth, so the "../" count is now wrong and the
            # path resolves to nowhere.
            #
            # Primary fix: search data_folder for a folder/file with the same
            # NAME (e.g. "concat_session") and use that instead.
            #
            # UCL patch (2nd tier): the primary basename search fails whenever
            # the raw data folder was staged under a different name than its
            # original basename -- e.g. Nextflow stages the raw session as
            # "ecephys_session" regardless of what it was called on disk when
            # preprocessing ran (e.g. "concat_session"), so no folder named
            # "concat_session" exists anywhere under data_folder to find.
            # When basename search fails, check whether this specific broken
            # path belongs to a raw recording *reader* node (identified via
            # the SpikeInterface JSON's "class" field, e.g.
            # "...SpikeGLXRecordingExtractor" / "...OpenEphysBinaryRecordingExtractor")
            # rather than some other unrelated folder_path (e.g. a whitening
            # matrix or motion folder). If so, and the caller has told us
            # where the raw session actually lives (ecephys_folder), use that
            # directly -- we already know precisely where it is, so there's
            # no need to guess by name.
            # UCL patch (round 3): concatenated/append_recordings sessions are
            # saved out as a plain binary-folder recording rather than read
            # back in via one of the format-specific readers above, so their
            # class name is "BinaryFolderRecording" / "BinaryRecordingExtractor"
            # instead of e.g. "SpikeGLXRecordingExtractor". Add those classes
            # by exact name (not a loose "Binary" substring match) since
            # intermediate binary-saved caches elsewhere in the preprocessing
            # chain can share the same class name -- a substring match would
            # risk mis-identifying one of those as the raw session again, the
            # same failure mode round 1 of this patch had to fix (see
            # UCL_README, saturation-count regression). [UNCONFIRMED]: verify
            # this doesn't regress on the non-concatenated SpikeGLX sessions
            # before promoting out of UNCONFIRMED.
            raw_reader_class_markers = (
                "SpikeGLX",
                "OpenEphys",
                "NeuroScope",
                "Neuralynx",
                "Intan",
                "BinaryFolderRecording",
                "BinaryRecordingExtractor",
            )

            def _is_raw_reader_class(class_name):
                return isinstance(class_name, str) and any(
                    marker in class_name for marker in raw_reader_class_markers
                )

            try:
                recording_dict = json.load(open(preprocessed_json_file))

                def _fix_paths(obj, class_name=None):
                    if isinstance(obj, dict):
                        # this dict is itself a SpikeInterface extractor node
                        # (has "class"/"kwargs"): remember its class name so
                        # that when we recurse into its "kwargs" dict, we know
                        # what kind of reader we're looking at.
                        node_class_name = obj.get("class", class_name)
                        for key in ("folder_path", "file_path"):
                            if key in obj and isinstance(obj[key], str):
                                candidate = (data_folder / obj[key]).resolve()
                                if not candidate.exists():
                                    basename = Path(obj[key]).name
                                    matches = list(data_folder.rglob(basename))
                                    matches = [m for m in matches if m.is_dir() or m.is_file()]
                                    if len(matches) == 1:
                                        logging.info(
                                            f"UCL patch: remapped broken path '{obj[key]}' "
                                            f"-> '{matches[0]}' (basename match)"
                                        )
                                        obj[key] = str(matches[0])
                                    elif (
                                        _is_raw_reader_class(class_name)
                                        and ecephys_folder is not None
                                        and Path(ecephys_folder).exists()
                                    ):
                                        logging.warning(
                                            f"UCL patch: could not remap broken path '{obj[key]}' "
                                            f"by basename (found {len(matches)} candidate(s)); node "
                                            f"class '{class_name}' looks like a raw recording reader, "
                                            f"so falling back to the known raw session mount "
                                            f"'{ecephys_folder}' instead."
                                        )
                                        obj[key] = str(ecephys_folder)
                                    else:
                                        logging.warning(
                                            f"UCL patch: could not remap broken path '{obj[key]}' "
                                            f"(basename '{basename}'): found {len(matches)} candidate(s) "
                                            f"under {data_folder}: {matches}"
                                        )
                        for v in obj.values():
                            _fix_paths(v, class_name=node_class_name)
                    elif isinstance(obj, list):
                        for v in obj:
                            _fix_paths(v, class_name=class_name)

                _fix_paths(recording_dict)
                recording_preprocessed = si.load(recording_dict, base_folder=data_folder)
            except Exception as e:
                logging.info(f"UCL patch: path-remap fallback also failed: {e}")
        if recording_preprocessed is None:
            logging.info("Error loading preprocessed data...")
    else:
        logging.info("Preprocessed recording not found...")
    return recording_preprocessed


def get_recording_relative_path(recording):
    """
    Returns path of recording file relative to session folder.

    If more than one file path is found, it returns None.
    """
    from spikeinterface.core.core_tools import _get_paths_list

    relative_path = None
    possible_substrings = ["ecephys/", "ecephys_compressed/", "ecephys_session/"]
    file_paths = _get_paths_list(recording.to_dict(recursive=True))

    if len(file_paths) == 1:
        file_path = str(file_paths[0])
        for possible_substring in possible_substrings:
            substring_index = str(file_path).find(possible_substring)
            if substring_index > 0:
                relative_path = file_path[substring_index:]
                break
    return relative_path


def get_default_curation_value(sorting_analyzer: si.SortingAnalyzer) -> dict:
    """
    Generate default curation value dict from sorting analyzer.

    Parameters
    ----------
    sorting_analyzer : si.SortingAnalyzer
        The sorting analyzer object.

    Returns
    -------
    curation_value : dict
        The default curation value dict.
    """
    # prepare the curation data using decoder labels
    curation_dict = default_curation_dict.copy()
    curation_dict["unit_ids"] = sorting_analyzer.unit_ids
    if "decoder_label" in sorting_analyzer.sorting.get_property_keys():
        decoder_labels = sorting_analyzer.get_sorting_property("decoder_label")
        noise_units = sorting_analyzer.unit_ids[decoder_labels == "noise"]
        curation_dict["removed"] = list(noise_units)
        for unit_id in noise_units:
            curation_dict["manual_labels"].append({"unit_id": unit_id, "quality": ["noise"]})
    try:
        validate_curation_dict(curation_dict)
        # make it serializable
        curation_dict = json.loads(json.dumps(check_json(curation_dict)))
    except ValueError as e:
        curation_dict = None
    return curation_dict


### METRICS GENERATION FUNCTIONS ###
def plot_raw_data(
    recording: si.BaseRecording,
    recording_lfp: si.BaseRecording | None = None,
    num_snippets_per_segment: int = 3,
    duration_s: float = 0.1,
    freq_ap: float = 300,
    freq_lfp: float = 500,
    channel_labels: np.ndarray | None = None
):
    """
    Plot snippets of raw data as an image

    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    recording_lfp : BaseRecording | None
        The LFP recording object.
    num_snippets_per_segment: int, default: 3
        Number of snippets to plot for each segment.
    duration_s : float, default: 0.1
        The duration of each snippet.
    freq_ap : float, default 300
        The highpass cutoff frequency for the ap band
    freq_lfp : float, default: 500
        The lowpass cutoff frequency in case recording_lfp is None
    channel_labels: np.ndarray | None
        The channel labels from preprocessing. Out, dead, noise

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    num_segments = recording.get_num_segments()
    fig, axs = _get_fig_axs(num_segments * 2, num_snippets_per_segment)

    recording_hp = spre.highpass_filter(recording, freq_min=freq_ap)
    if recording_lfp is None:
        recording_lfp = spre.bandpass_filter(recording, freq_min=0.1, freq_max=freq_lfp)
    
    surface_channel_y_position = _get_surface_channel(recording, channel_labels)

    for segment_index in range(num_segments):
        # evenly distribute t_starts across segments
        times = recording.get_times(segment_index=segment_index)
        t_starts = np.round(np.linspace(times[0], times[-1], num_snippets_per_segment + 2)[1:-1], 1)
        for snippet_index, t_start in enumerate(t_starts):
            ax_ap = axs[segment_index * 2, snippet_index]
            ax_lfp = axs[segment_index * 2 + 1, snippet_index]
            sw.plot_traces(
                recording_hp,
                time_range=[t_start, t_start + duration_s],
                segment_index=segment_index,
                mode="map",
                return_scaled=True,
                with_colorbar=True,
                ax=ax_ap,
                clim=(-50, 50),
            )
            sw.plot_traces(
                recording_lfp,
                time_range=[t_start, t_start + duration_s],
                segment_index=segment_index,
                mode="map",
                return_scaled=True,
                with_colorbar=True,
                ax=ax_lfp,
                clim=(-300, 300),
            )
            if surface_channel_y_position is not None:
                ax_ap.axhline(y=surface_channel_y_position, c='g')
                ax_lfp.axhline(y=surface_channel_y_position, c='g')

            if np.mod(num_snippets_per_segment, 2) == 1:
                if snippet_index == num_snippets_per_segment // 2:
                    ax_ap.set_title(f"seg{segment_index} @ {t_start}s\nAP")
                    ax_lfp.set_title(f"LFP")
                else:
                    ax_ap.set_title(f"seg{segment_index} @ {t_start}s\n ")
            else:
                ax_ap.set_title(f"seg{segment_index} @ {t_start}s\nAP")
                ax_lfp.set_title(f"LFP")
            if snippet_index == 0:
                ax_ap.set_ylabel("Depth ($\mu m$)")
                ax_lfp.set_ylabel("Depth ($\mu m$)")
            else:
                ax_ap.set_yticklabels([])
                ax_lfp.set_yticklabels([])
            if segment_index == num_segments - 1:
                ax_lfp.set_xlabel(f"Time (tot. {duration_s} s)")
            ax_ap.set_xticklabels([])
            ax_lfp.set_xticklabels([])

    fig.subplots_adjust(wspace=0.3, hspace=0.2, top=0.9)
    return fig


def plot_psd(
    recording: si.BaseRecording,
    num_snippets_per_segment: int = 3,
    duration_s: float = 1,
    channel_labels: np.ndarray | None = None
):
    """
    Plot spectra for wide-band.

    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    num_snippets_per_segment: int, default: 3
        Number of snippets to plot for each segment.
    duration_s : float, default: 1
        The duration of each snippet.
    channel_labels: np.ndarray | None
        The channel labels from preprocessing. Out, dead, noise

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the wide-band frequency spectrum.
    """
    num_segments = recording.get_num_segments()
    fig_psd, axs_psd = _get_fig_axs(num_segments * 2, num_snippets_per_segment)

    depths = recording.get_channel_locations()[:, 1]

    surface_channel_y_position = _get_surface_channel(recording, channel_labels)

    for segment_index in range(num_segments):
        # evenly distribute t_starts across segments
        times = recording.get_times(segment_index=segment_index)
        t_starts = np.round(np.linspace(times[0], times[-1], num_snippets_per_segment + 2)[1:-1], 1)
        for snippet_index, t_start in enumerate(t_starts):
            ax_psd = axs_psd[segment_index * 2, snippet_index]
            ax_psd_channels = axs_psd[segment_index * 2 + 1, snippet_index]

            start_frame = recording.time_to_sample_index(t_start, segment_index=segment_index)
            end_frame = recording.time_to_sample_index(t_start + duration_s, segment_index=segment_index)
            traces = recording.get_traces(
                start_frame=start_frame, end_frame=end_frame, segment_index=segment_index, return_scaled=True
            )

            power_channels = []

            for i in range(traces.shape[1]):
                f, p = welch(traces[:, i], fs=recording.sampling_frequency)
                power_channels.append(p)
                ax_psd.plot(f, p, color="gray", alpha=0.5)

            power_channels = np.array(power_channels)

            p_mean = np.mean(power_channels, axis=0)
            ax_psd.plot(f, p_mean, color="k", lw=1)

            extent = [f.min(), f.max(), np.min(depths), np.max(depths)]
            ax_psd_channels.imshow(
                power_channels, extent=extent, aspect="auto", cmap="inferno", origin="lower", norm="log"
            )
            ax_psd.set_title(f"seg{segment_index} @ {t_start}s")
            if snippet_index == 0:
                ax_psd.set_ylabel("Power ($\mu V^2/Hz$)")
                ax_psd_channels.set_ylabel("Depth ($\mu$ m)")
            if segment_index == num_segments - 1:
                ax_psd_channels.set_xlabel("Frequency (Hz)")

            ax_psd.set_yscale("log")

            if surface_channel_y_position is not None:
                ax_psd_channels.axhline(y=surface_channel_y_position, c='g')

    fig_psd.subplots_adjust(wspace=0.3, hspace=0.3, top=0.8)
    fig_psd.suptitle("PSD - Wideband")

    return fig_psd


def plot_rms_by_depth(recording, recording_preprocessed=None, recording_lfp=None, channel_labels: np.ndarray | None = None):
    """ """
    num_segments = recording.get_num_segments()
    fig_rms, ax_rms = plt.subplots(figsize=(5, 8))

    surface_channel_y_position = _get_surface_channel(recording, channel_labels)

    if recording_lfp is None:
        # this means the recording is wide-band, so we apply an additional hp filter
        recording = spre.highpass_filter(recording)

    recording = spre.average_across_direction(recording, direction="y")

    data_raw = si.get_random_data_chunks(recording, return_scaled=True)
    depths_raw = recording.get_channel_locations()[:, 1]
    rms_raw = np.sqrt(np.sum(data_raw**2, axis=0) / data_raw.shape[0])

    ax_rms.plot(rms_raw, depths_raw, color="gray", label="raw")

    if recording_preprocessed is not None:
        recording_preprocessed = spre.average_across_direction(recording_preprocessed, direction="y")
        data_pre = si.get_random_data_chunks(recording_preprocessed, return_scaled=True)

        depths_pre = recording_preprocessed.get_channel_locations()[:, 1]
        rms_pre = np.sqrt(np.sum(data_pre**2, axis=0) / data_pre.shape[0])
        ax_rms.plot(rms_pre, depths_pre, color="r", label="preprocessed")
        ax_rms.legend()

    if surface_channel_y_position is not None:
        ax_rms.axhline(y=surface_channel_y_position, c='g')

    ax_rms.set_xlabel("RMS ($\mu V$)")
    ax_rms.set_ylabel("Depth ($\mu m$)")
    ax_rms.spines[["right", "top"]].set_visible(False)

    fig_rms.subplots_adjust(top=0.8)
    return fig_rms, ax_rms


def generate_raw_qc(
    recording: si.BaseRecording,
    recording_name: str,
    output_qc_path: Path,
    relative_to: Path | None = None,
    recording_lfp: si.BaseRecording | None = None,
    recording_preprocessed: si.BaseRecording | None = None,
    processing: Processing | None = None,
    visualization_output: dict | None = None,
    allow_failed_metrics: bool = False
) -> dict[str : list[QCMetric]]:
    """
    Generate raw data quality control metrics for a given recording.

    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    recording_name : str
        The name of the recording.
    output_qc_path : Path
        The output path for the quality control.
    relative_to : Path | None, default: None
        The relative path to the output path.
    recording_lfp : BaseRecording | None, default: None
        The LFP recording object.
    recording_preprocessed : BaseRecording | None, default: None
        The preprocessed recording object.
    processing : Processing | None, default: None
        The processing metadata object.
    visualization_output : dict | None, default: None
        The visualization output dict.

    Returns
    -------
    metrics : list[QCMetric]:
        The quality control metrics.
    """
    metrics = []
    recording_fig_folder = output_qc_path
    recording_fig_folder.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    status_pending = QCStatus(evaluator="", status=Status.PENDING, timestamp=now)
    status_pass = QCStatus(evaluator="", status=Status.PASS, timestamp=now)
    recording_name_abbrv = recording_abbrv_name(recording_name)

    channel_labels = None
    if processing is not None:
        try:
            data_processes = processing.processing_pipeline.data_processes
            for data_process in data_processes:
                params = data_process.parameters.model_dump()
                outputs = data_process.outputs.model_dump()
                if (
                    data_process.name == "Ephys preprocessing"
                    and params.get("recording_name") is not None
                    and params.get("recording_name") == recording_name
                ):
                    channel_labels = np.array(outputs.get("channel_labels"))
        except:
            logging.info(f"Failed to load bad channel labels for {recording_name}")

    logging.info("Generating RAW DATA metrics")
    fig_raw = plot_raw_data(recording, recording_lfp, channel_labels=channel_labels)
    raw_traces_path = recording_fig_folder / "traces_raw.png"
    fig_raw.savefig(raw_traces_path, dpi=300)
    if relative_to is not None:
        raw_traces_path = raw_traces_path.relative_to(relative_to)

    value_with_options = {
        "value": "",
        "options": ["Normal", "No spikes", "Noisy"],
        "status": ["Pass", "Fail", "Fail"],
        "type": "dropdown",
    }
    value = DropdownMetric(**value_with_options)
    if visualization_output is not None:
        if recording_name in visualization_output:
            timeseries_link = visualization_output[recording_name].get("timeseries")
            timeseries_str = f"[Sortingview]({timeseries_link})"
        else:
            timeseries_str = None
    else:
        timeseries_str = None

    raw_metric_description = (
        "This metric displays raw data snippets for both for AP and LFP streams. "
        "Check out the snippets for abnormal noise or lack of spikes."
    )
    if timeseries_str is not None:
        raw_metric_description += f"Raw data can also be visualized in {timeseries_str}"

    raw_data_metric = QCMetric(
        name=f"Raw data {recording_name_abbrv}",
        modality=Modality.ECEPHYS,
        stage=Stage.RAW,
        description=raw_metric_description,
        value=value,
        reference=str(raw_traces_path),
        status_history=[status_pending],
        tags={
            "probe": recording_name_abbrv
        }
    )
    metrics.append(raw_data_metric)

    logging.info("Generating PSD metrics")
    fig_psd = plot_psd(recording, channel_labels=channel_labels)
    psd_path = recording_fig_folder / "psd.png"

    fig_psd.savefig(psd_path, dpi=300)

    if relative_to is not None:
        psd_path = psd_path.relative_to(relative_to)

    value_with_options = {
        "value": "",
        "options": ["No contamination", "High frequency contamination", "Line contamination"],
        "status": ["Pass", "Fail", "Fail"],
        "type": "dropdown",
    }
    value = DropdownMetric(**value_with_options)
    psd_metric_description = (
        "This metric displays the power spectral density (PSD) of raw data. "
        "PDSs are computed by channel and shown as lines or map (by channel depth). "
        "Use the plots to spot contamination from high-frequency or line noise sources."
    )
    psd_metric = QCMetric(
        name=f"PSD {recording_name_abbrv}",
        modality=Modality.ECEPHYS,
        stage=Stage.RAW,
        description=psd_metric_description,
        reference=str(psd_path),
        value=None,
        status_history=[status_pass],
        tags={
            "probe": recording_name_abbrv
        }
    )
    metrics.append(psd_metric)

    logging.info("Generating NOISE metrics")
    fig_rms, ax_rms = plot_rms_by_depth(recording, recording_preprocessed, channel_labels=channel_labels)
    # Bad channel detection out of brain, noisy, silent
    if channel_labels is not None:
        metric_values = {
            "good": int(np.sum(channel_labels == "good")),
            "noise": int(np.sum(channel_labels == "noise")),
            "dead": int(np.sum(channel_labels == "dead")),
            "out": int(np.sum(channel_labels == "out")),
        }
        metric_values_str = None
        for metric_name, metric_value in metric_values.items():
            if metric_values_str is None:
                metric_values_str = f"{metric_name}: {metric_value}"
            else:
                metric_values_str += f"\n{metric_name}: {metric_value}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax_rms.text(
            0.2,
            0.9,
            metric_values_str,
            transform=fig_rms.transFigure,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

    rms_path = recording_fig_folder / "rms.png"
    fig_rms.savefig(rms_path, dpi=300)
    if relative_to is not None:
        rms_path = rms_path.relative_to(relative_to)

    # make metric for qc json
    value_with_options = {
        "value": "",
        "options": ["Good", "No out channels detected"],
        "status": ["Pass", "Fail"],
        "type": "dropdown",
    }
    value = DropdownMetric(**value_with_options)
    rms_mtric_description = (
        "This metric shows the noise levels (RMS) by depth for each channel, computed from the raw and "
        "preprocessed data. It also includes a summary of channel labels (after bad channel detection). "
        "Use this metric to check if out-of-brain channels are visible (lower RMS values "
        "at the top of the probe), but not labeled as 'out'."
    )
    rms_metric = QCMetric(
        name=f"RMS {recording_name_abbrv}",
        modality=Modality.ECEPHYS,
        stage=Stage.RAW,
        description=rms_mtric_description,
        reference=str(rms_path),
        value=value,
        status_history=[status_pass],
        tags={
            "probe": recording_name_abbrv
        }
    )
    metrics.append(rms_metric)

    return metrics


def generate_drift_qc(
    recording: si.BaseRecording,
    recording_name: str,
    motion_path: Path,
    output_qc_path: Path,
    relative_to: Path | None = None,
) -> QCMetric:
    """
    Generate drift metrics for a given sorting result.

    Parameters
    ----------
    recording : BaseRecording
        The sorting analyzer.
    recording_name : str
        The name of the recording.
    motion_path: Path,
        The path of the recording's motion folder.
    output_qc_path : Path
        The output path for the quality control.
    processing : Processing | None, default: None
        The processing metadata object.

    Returns
    -------
    metrics : list[QCMetric]:
        The quality control metrics.
    """

    logging.info("Generating DRIFT metric")

    recording_fig_folder = output_qc_path
    recording_fig_folder.mkdir(parents=True, exist_ok=True)
    recording_name_abbrv = recording_abbrv_name(recording_name)

    motion_info = spre.load_motion_info(motion_path)
    all_peaks = motion_info["peaks"]
    all_peak_locations = motion_info["peak_locations"]
    motion = motion_info["motion"]

    # UCL patch: for multi-segment recordings, the AIND preprocessing capsule
    # concatenates all segments into one block before calling compute_motion(),
    # because SpikeInterface's estimate_motion() does not yet support multi-segment
    # recordings natively (see AllenNeuralDynamics/spikeinterface#2626). As a result
    # `motion.displacement` (and `all_peaks["segment_index"]`) may only contain a
    # single segment (index 0) even when `recording` genuinely has multiple segments
    # (e.g. several days of SpikeGLX concatenated via si.append_recordings()).
    #
    # Looping per-segment against `recording.get_num_segments()` in that case finds
    # no peaks for segment_index >= 1 and crashes. Instead, when motion was computed
    # on a concatenated view, we plot one continuous raster across the full
    # concatenated timeline (which is what was actually estimated) and mark the real
    # segment boundaries so inter-segment drift jumps -- the thing this plot exists to
    # catch -- stay visible, rather than splitting into segment panels that don't
    # correspond to what was computed.
    motion_is_concatenated = (
        motion is not None
        and len(motion.displacement) == 1
        and recording.get_num_segments() > 1
    )

    if motion_is_concatenated:
        num_plot_segments = 1
        # cumulative sample-boundary offsets (in seconds) between real segments,
        # for marking day/session boundaries on the concatenated raster
        segment_boundaries_s = []
        cumulative_samples = 0
        for s in range(recording.get_num_segments() - 1):
            cumulative_samples += recording.get_num_samples(s)
            segment_boundaries_s.append(cumulative_samples / recording.sampling_frequency)

        # UCL patch: the boundary markers above assume the raw `recording`'s
        # per-segment sample counts and order are the same ones that were
        # concatenated to produce `motion`. That should hold as long as
        # preprocessing doesn't drop/reorder samples, but if it ever doesn't
        # (e.g. a future change to how segments are concatenated upstream),
        # the markers would be silently wrong rather than absent. Check the
        # raw recording's total duration against the actual duration covered
        # by the concatenated motion estimate and warn loudly if they disagree.
        total_duration_from_recording_s = (
            sum(recording.get_num_samples(s) for s in range(recording.get_num_segments()))
            / recording.sampling_frequency
        )
        total_duration_from_motion_s = motion.temporal_bins_s[0][-1]
        duration_mismatch_s = abs(total_duration_from_recording_s - total_duration_from_motion_s)
        # allow slack for motion's temporal binning (bins are centered, not
        # edge-aligned, so the last bin center is expected to fall short of
        # the true end by roughly half a bin width -- a few seconds is a
        # generous margin for the bin widths typically used here)
        duration_tolerance_s = 5.0
        if duration_mismatch_s > duration_tolerance_s:
            logging.warning(
                f"UCL patch: segment-boundary markers may be WRONG for {recording_name}. "
                f"Raw recording total duration ({total_duration_from_recording_s:.1f}s) does not "
                f"match the duration covered by the concatenated motion estimate "
                f"({total_duration_from_motion_s:.1f}s), difference={duration_mismatch_s:.1f}s. "
                f"This likely means `recording`'s segments are not the same segments that were "
                f"concatenated in preprocessing (different sample counts, order, or a segment was "
                f"dropped/added) -- do not trust the blue boundary lines on this drift plot."
            )
    else:
        num_plot_segments = recording.get_num_segments()
        segment_boundaries_s = []

    fig_drift, axs_drift = plt.subplots(ncols=num_plot_segments, figsize=(10, 10))
    y_locs = recording.get_channel_locations()[:, 1]
    sampling_frequency = recording.sampling_frequency
    depth_lim = [np.min(y_locs), np.max(y_locs)]

    max_displacement = None
    depth_at_max_displacement = None
    max_cumulative_drift = None
    depth_at_max_cumulative_drift = None

    for plot_index in range(num_plot_segments):
        if num_plot_segments == 1:
            ax_drift = axs_drift
        else:
            ax_drift = axs_drift[plot_index]

        if motion_is_concatenated:
            # motion/peaks are already expressed in concatenated-timeline terms:
            # plot everything against the single (index 0) motion segment.
            peaks_to_plot = all_peaks
            peak_locations_to_plot = all_peak_locations
            motion_segment_index = 0
        else:
            segment_mask = all_peaks["segment_index"] == plot_index
            peaks_to_plot = all_peaks[segment_mask]
            peak_locations_to_plot = all_peak_locations[segment_mask]
            motion_segment_index = plot_index

        _ = sw.plot_drift_raster_map(
            sorting_analyzer=None,
            peaks=peaks_to_plot,
            peak_locations=peak_locations_to_plot,
            recording=recording,
            sampling_frequency=sampling_frequency,
            segment_index=motion_segment_index,
            depth_lim=depth_lim,
            clim=(-200, 0),
            cmap="Greys_r",
            scatter_decimate=10,
            alpha=0.3,
            ax=ax_drift,
        )
        ax_drift.spines["top"].set_visible(False)
        ax_drift.spines["right"].set_visible(False)

        if motion_is_concatenated:
            for boundary_s in segment_boundaries_s:
                ax_drift.axvline(boundary_s, color="blue", linestyle="--", linewidth=1, alpha=0.7)

        if motion is not None:
            displacement_arr = motion.displacement[motion_segment_index]
            temporal_bins = motion.temporal_bins_s[motion_segment_index]
            spatial_bins = motion.spatial_bins_um

            # calculate cumulative_drift and max displacement
            drift_ptps = np.ptp(displacement_arr, axis=0)
            displacements_diff_arr = np.diff(displacement_arr, axis=0)
            cumulative_drifts = np.sum(displacements_diff_arr, axis=0)
            max_displacement_index = np.argmax(drift_ptps)
            max_displacement = np.round(drift_ptps[max_displacement_index], 2)
            depth_at_max_displacement = int(spatial_bins[max_displacement_index])

            max_cumulative_drift_index = np.argmax(cumulative_drifts)
            max_cumulative_drift = np.round(cumulative_drifts[max_cumulative_drift_index], 2)
            depth_at_max_cumulative_drift = int(spatial_bins[max_cumulative_drift_index])

            ax_drift.plot(temporal_bins, displacement_arr + spatial_bins, color="red", alpha=0.5)

    if motion is not None:
        title = (
            f"Max displacement: {max_displacement} $\mu m$ (depth: {depth_at_max_displacement} ) $\\mu m$\n"
            f"Max cumulative drift: {max_cumulative_drift} $\mu m$ (depth: {depth_at_max_cumulative_drift} ) $\\mu m$\n"
        )
        if motion_is_concatenated:
            title += (
                f"UCL note: motion estimated on {recording.get_num_segments()} concatenated segments; "
                f"blue dashed lines mark segment boundaries at {[round(b, 1) for b in segment_boundaries_s]}s\n"
            )
        ax_drift.set_title(title)

    drift_map_path = recording_fig_folder / "drift_map.png"
    fig_drift.savefig(drift_map_path, dpi=300)
    if relative_to is not None:
        drift_map_path = drift_map_path.relative_to(relative_to)

    # make metric for qc json
    value_with_options = {
        "value": "",
        "options": ["Good", "High drift", "Bad drift estimation"],
        "status": ["Pass", "Fail", "Fail"],
        "type": "dropdown",
    }
    value = DropdownMetric(**value_with_options)
    drift_metric_description = (
        "This metric shows a rastermap of the recording with overlaid estimated motion. "
        "Use this metric to spot high drifts or bad drift estimation."
    )
    drift_metric = QCMetric(
        name=f"Probe Drift - {recording_name_abbrv}",
        modality=Modality.ECEPHYS,
        stage=Stage.RAW,
        description=drift_metric_description,
        reference=str(drift_map_path),
        value=value,
        status_history=[QCStatus(evaluator="", status=Status.PENDING, timestamp=datetime.now())],
        tags={
            "probe": recording_name_abbrv
        }
        
    )
    drift_metrics = [drift_metric]

    return drift_metrics


def generate_event_qc(
    recording: si.BaseRecording,
    recording_name: str,
    output_qc_path: Path,
    relative_to: Path | None = None,
    event_dict: dict | None = None,
    event_keys: list[str] = ["licktime", "optogeneticstime"],
    **job_kwargs,
):
    """
    Generate event metrics for a given recording.
    The event metrics include saturation and responses to certain events.

    Returns
    -------
    metrics : list[QCMetric]:
        The quality control metrics.
    """
    recording_fig_folder = output_qc_path
    recording_fig_folder.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    status_pending = QCStatus(evaluator="", status=Status.PENDING, timestamp=now)
    status_pass = QCStatus(evaluator="", status=Status.PASS, timestamp=now)
    recording_name_abbrv = recording_abbrv_name(recording_name)

    logging.info("Generating SATURATION metric")

    probes_info = recording.get_annotation("probes_info")
    if probes_info is None:
        saturation_threshold_uv = None
    else:
        part_number = recording.get_annotation("probes_info")[0].get("part_number")
        saturation_threshold_uv = saturation_thresholds_uv.get(part_number)

    metrics = []
    if saturation_threshold_uv is None:
        logging.info(
            f"\tSaturation threshold for {recording_name} not available. "
            "Cannot generate saturation metrics."
        )
    else:
        pos_evts, neg_evts = find_saturation_events(recording, saturation_threshold_uv, **job_kwargs)

        if len(pos_evts) > 0:
            logging.info(f"\tFound {len(pos_evts)} positive saturation events!")
        else:
            logging.info("\tNo positive saturation events found")
        if len(neg_evts) > 0:
            logging.info(f"\tFound {len(neg_evts)} negative saturation events!")
        else:
            logging.info("\tNo negative saturation events found")

        # saturation events timeline
        fig_sat_time, ax_sat_time = plt.subplots(figsize=(15, 10))
        ax_sat_time.ticklabel_format(useOffset=False, style='plain', axis='x')

        ax_sat_time.set_title(
            f"Saturation events:\nPositive: {len(pos_evts)} -- Negative: {len(neg_evts)}"
        )
        def _get_event_times(recording, evts):
            """
            UCL patch (why we're here): recording.get_times() errors out on
            multi-segment recordings (e.g. multi-day sessions built with
            si.append_recordings) unless you tell it which segment.

            Fix: each event already knows which segment it's from
            (evts["segment_index"]), so look up times per-segment instead
            of calling get_times() on the whole recording at once.
            """
            if recording.get_num_segments() == 1:
                return recording.get_times()[evts["sample_index"]]
            evt_times = np.empty(len(evts), dtype=float)
            for seg_index in np.unique(evts["segment_index"]):
                seg_mask = evts["segment_index"] == seg_index
                seg_times = recording.get_times(segment_index=int(seg_index))
                evt_times[seg_mask] = seg_times[evts["sample_index"][seg_mask]]
            return evt_times

        if len(pos_evts) > 0:
            pos_evt_times = _get_event_times(recording, pos_evts)
            ax_sat_time.plot(pos_evt_times, np.ones_like(pos_evt_times), ls="", marker="|", markersize=20, color="r", label="positive")
        if len(neg_evts) > 0:
            neg_evt_times = _get_event_times(recording, neg_evts)
            ax_sat_time.plot(neg_evt_times, -np.ones_like(neg_evt_times), ls="", marker="|", markersize=20, color="b", label="negative")
        ax_sat_time.legend()
        ax_sat_time.set_xlabel("Time (s)")
        ax_sat_time.set_ylabel("Sign")
        ax_sat_time.set_yticks([-1, +1])
        ax_sat_time.set_yticklabels(["-", "+"])
        ax_sat_time.set_ylim(-2, 2)
        ax_sat_time.spines[["top", "right"]].set_visible(False)
        ax_sat_time.get_xaxis().get_major_formatter().set_scientific(False)

        fig_sat_time_path = recording_fig_folder / "saturation_timeline.png"
        fig_sat_time.savefig(fig_sat_time_path, dpi=300)
        if relative_to is not None:
            fig_sat_time_path = fig_sat_time_path.relative_to(relative_to)

        if len(pos_evts) > 0 or len(neg_evts) > 0:
            value_with_options = {
                "value": "",
                "options": ["Good", "Too many saturation events"],
                "status": ["Pass", "Fail"],
                "type": "dropdown",
            }
            value = DropdownMetric(**value_with_options)
            saturation_status = status_pending
        else:
            value = {"value": "Pass"}
            saturation_status = status_pass

        saturation_timeline_metric_description = (
            "This metric shows a scatter plot of saturation events, with red and blue markers for positive and negative "
            "events, respectively. Use this metric to spot an abnormal number of saturation events."
        )
        saturation_timeline_metric = QCMetric(
            name=f"Saturation events timeline {recording_name_abbrv}",
            modality=Modality.ECEPHYS,
            stage=Stage.RAW,
            description=saturation_timeline_metric_description,
            reference=str(fig_sat_time_path),
            value=value,
            status_history=[saturation_status],
            tags={
                "probe": recording_name_abbrv
            }
        )

        metrics.append(saturation_timeline_metric)

    if event_dict is not None:
        logging.info("Generating TRIGGER EVENT metrics")

        # make a sorting object with the events
        unit_dict = {}
        for k in event_dict.keys():
            if any(keyword in k.lower() for keyword in event_keys):
                events = np.array(event_dict[k])
                events_in_range = events[events >= recording.get_times()[0]]
                events_in_range = events_in_range[events_in_range < recording.get_times()[-1]]
                if len(events_in_range) > 0:
                    sample_indices = recording.time_to_sample_index(events_in_range)
                    unit_dict[k] = sample_indices
        if len(unit_dict) > 0:
            logging.info(f"\tFound {len(unit_dict)} trigger event sources for {event_keys} keywords.")
            for event_key, event_values in unit_dict.items():
                logging.info(f"\t{event_key}: {len(event_values)} events")
            sorting_events = si.NumpySorting.from_unit_dict(
                [unit_dict], sampling_frequency=recording.sampling_frequency
            )
            analyzer = si.create_sorting_analyzer(sorting_events, recording, sparse=False)

            extensions = {
                "random_spikes": {"method": "uniform"},
                "waveforms": {"ms_before": 10, "ms_after": 10},
                "templates": {"ms_before": 10, "ms_after": 10},
            }
            analyzer.compute(extensions, **job_kwargs)
            template_ext = analyzer.get_extension("templates")
            templates = template_ext.get_data()

            # create plots for each event
            fig_events, axs_events = _get_fig_axs(ncols=1, nrows=len(analyzer.unit_ids))

            num_spikes = analyzer.sorting.count_num_spikes_per_unit()

            for unit_index, unit_id in enumerate(analyzer.unit_ids):
                ax = axs_events[unit_index, 0]
                # color by depth
                cmap = plt.get_cmap("viridis", analyzer.get_num_channels())
                depths = analyzer.get_channel_locations()[:, 1]
                norm = mpl.colors.Normalize(vmin=np.min(depths), vmax=np.max(depths))

                colors = cmap(np.arange(analyzer.get_num_channels()))
                times = np.linspace(
                    -template_ext.params["ms_before"], template_ext.params["ms_after"], templates.shape[1]
                )

                for template, color in zip(templates[unit_index].T, colors):
                    ax.plot(times, template, color=color, alpha=0.3)

                ax.set_title(f"{unit_id} (#{num_spikes[unit_id]})")
                ax.set_ylabel("Voltage ($\\mu V$)")
                if unit_index == analyzer.get_num_units() - 1:
                    ax.set_xlabel("Time ($ms$)")
                ax.spines[["top", "right"]].set_visible(False)
                fig_events.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="vertical", label="Depth ($\\mu m$)"
                )
            fig_events.subplots_adjust(hspace=0.3, left=0.3, right=0.85)

            value_with_options = {
                "value": "",
                "options": ["Good", "Trigger artifacts"],
                "status": ["Pass", "Fail"],
                "type": "dropdown",
            }
            value = DropdownMetric(**value_with_options)
            events_status = status_pending
        else:
            logging.info(f"\tNo events found for {recording_name}")

            fig_events = None

        # TODO: maybe add raw / preprocessed?
        if fig_events is not None:
            fig_events_path = recording_fig_folder / "trigger_events.png"
            fig_events.savefig(fig_events_path, dpi=300)
            if relative_to is not None:
                fig_events_path = fig_events_path.relative_to(relative_to)

            trigger_event_metric_description  = (
                "This metric shows the average signals correpsinding to trigger events that could cause "
                "artifacts in the traces (e.g., opto stimulation, lick events). Use this metric to report "
                "abnormal artifacts that might affect spike sorting."
            )
            trigger_event_metric = QCMetric(
                name=f"Trigger events {recording_name_abbrv}",
                modality=Modality.ECEPHYS,
                stage=Stage.RAW,
                description=trigger_event_metric_description,
                reference=str(fig_events_path),
                value=value,
                status_history=[events_status],
                tags={
            "probe": recording_name_abbrv
        }
            )
            metrics.append(trigger_event_metric)

    return metrics


def generate_units_qc(
    sorting_analyzer: si.SortingAnalyzer | None,
    recording_name: str,
    output_qc_path: Path,
    relative_to: Path | None = None,
    visualization_output: dict | None = None,
    raw_recording: si.BaseRecording | None = None,
    max_amplitude_for_visualization: float = 5000,
    max_firing_rate_for_visualization: float = 50,
    bin_duration_hist_s: float = 1,
) -> dict[str : list[QCMetric]]:
    """
    Generate unit yield metrics for a given sorting result.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer.
    recording_name : str
        The name of the recording.
    output_qc_path : Path
        The output path for the quality control.
    relative_to : Path | None, default: None
        The relative path to the output path.
    visualization_output : dict | None, default: None
        The visualization output dict.
    raw_recording : BaseRecording
        The raw recording (used to infer recording path)
    max_amplitude_for_visualization : float, default: 5000
        The maximum amplitude used for plotting.
    max_firing_rate_for_visualization : float, default: 50
        The maximum firing rate used for plotting.
    bin_duration_hist_s : float, default: 1
        The duration of the histogram bins.

    Returns
    -------
    metrics : list[QCMetric]:
        The quality control metrics.
    """    
    metrics = []
    recording_fig_folder = output_qc_path
    recording_fig_folder.mkdir(exist_ok=True, parents=True)
    now = datetime.now()
    status_pending = QCStatus(evaluator="", status=Status.PENDING, timestamp=now)
    recording_name_abbrv = recording_abbrv_name(recording_name)

    logging.info("Generating UNIT YIELD metric")
    if sorting_analyzer is None:
        logging.info(f"\tNo sorting analyzer found for {recording_name}")
        return metrics

    number_of_units = sorting_analyzer.get_num_units()

    decoder_label = sorting_analyzer.sorting.get_property("decoder_label")
    number_of_sua_units = int(np.sum(decoder_label == "sua"))
    number_of_mua_units = int(np.sum(decoder_label == "mua"))
    number_of_noise_units = int(np.sum(decoder_label == "noise"))

    default_qc = sorting_analyzer.sorting.get_property("default_qc")
    number_of_good_units = int(np.sum(default_qc == True))

    quality_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()
    template_metrics = sorting_analyzer.get_extension("template_metrics").get_data()

    fig_yield, axs_yield = plt.subplots(3, 3, figsize=(15, 10))

    # protect against all NaNs
    bins = np.linspace(0, 2, 20)
    ax_isi = axs_yield[0, 0]
    if not np.isnan(quality_metrics["isi_violations_ratio"]).all():
        ax_isi.hist(quality_metrics["isi_violations_ratio"], bins=bins, density=True)
    ax_isi.set_xscale("log")
    ax_isi.set_title(f"ISI Violations Ratio")
    ax_isi.spines[["top", "right"]].set_visible(False)

    ax_amp_cutoff = axs_yield[0, 1]
    if not np.isnan(quality_metrics["amplitude_cutoff"]).all():
        ax_amp_cutoff.hist(quality_metrics["amplitude_cutoff"], bins=20, density=True)
    ax_amp_cutoff.set_title(f"Amplitude Cutoff")
    ax_amp_cutoff.spines[["top", "right"]].set_visible(False)

    ax_presence_ratio = axs_yield[0, 2]
    if not np.isnan(quality_metrics["presence_ratio"]).all():
        ax_presence_ratio.hist(quality_metrics["presence_ratio"], bins=20, density=True)
    ax_presence_ratio.set_title(f"Presence Ratio")
    ax_presence_ratio.spines[["top", "right"]].set_visible(False)

    ax_drift = axs_yield[1, 0]
    if not np.isnan(quality_metrics['drift_ptp']).all():
        ax_drift.hist(quality_metrics['drift_ptp'], bins=20, density=True)
    ax_drift.set_title(f"Drift Peak to Peak")
    ax_drift.spines[["top", "right"]].set_visible(False)

    ax_snr = axs_yield[1, 1]
    if not np.isnan(quality_metrics['snr']).all():
        ax_snr.hist(quality_metrics['snr'], bins=20, density=True)
    ax_snr.set_title(f"SNR")
    ax_snr.spines[["top", "right"]].set_visible(False)

    ax_halfwidth = axs_yield[1, 2]
    if not np.isnan(template_metrics['half_width']).all():
        ax_halfwidth.hist(template_metrics['half_width'], bins=20, density=True)
    ax_halfwidth.set_title(f"Half Width")
    ax_halfwidth.spines[["top", "right"]].set_visible(False)

    channel_indices = np.array(
        list(si.get_template_extremum_channel(sorting_analyzer, mode="peak_to_peak", outputs="index").values())
    )
    channel_depths = sorting_analyzer.get_channel_locations()[channel_indices, 1]
    amplitudes = np.array(list(si.get_template_extremum_amplitude(sorting_analyzer, mode="peak_to_peak").values()))
    amplitudes[amplitudes > max_amplitude_for_visualization] = max_amplitude_for_visualization
    df_amplitudes_depths = pd.DataFrame({"amplitude": amplitudes, "channel_depth": channel_depths})
    mean_amplitude_by_depth = df_amplitudes_depths.groupby("channel_depth").mean()

    colors = {"sua": "green", "mua": "orange", "noise": "red"}
    ax_amplitudes = axs_yield[2, 0]
    if decoder_label is not None:
        for label in np.unique(decoder_label):
            mask = decoder_label == label
            ax_amplitudes.scatter(amplitudes[mask], channel_depths[mask], c=colors[label], label=label, alpha=0.4)
    else:
        ax_amplitudes.scatter(amplitudes, channel_depths, alpha=0.4)

    try:
        smoothed_amplitude = savgol_filter(mean_amplitude_by_depth["amplitude"], 10, 2)
        ax_amplitudes.plot(smoothed_amplitude, mean_amplitude_by_depth.index.tolist(), c="r")
    except Exception:
        logging.info("Smooting amplitudes failed.")
    ax_amplitudes.set_title("Unit Amplitude By Depth")
    ax_amplitudes.set_xlabel("Amplitude ($\\mu V$)")
    ax_amplitudes.set_ylabel("Depth ($\\mu m$)")
    ax_amplitudes.legend()
    ax_amplitudes.spines[["top", "right"]].set_visible(False)

    ax_fr = axs_yield[2, 1]
    firing_rate = np.array(quality_metrics["firing_rate"].tolist())
    firing_rate[firing_rate > max_firing_rate_for_visualization] = max_firing_rate_for_visualization
    df_firing_rate_depths = pd.DataFrame({"firing_rate": firing_rate, "channel_depth": channel_depths})
    mean_firing_rate_by_depth = df_firing_rate_depths.groupby("channel_depth").mean()

    if decoder_label is not None:
        for label in np.unique(decoder_label):
            mask = decoder_label == label
            ax_fr.scatter(firing_rate[mask], channel_depths[mask], c=colors[label], label=label, alpha=0.4)
    else:
        ax_fr.scatter(firing_rate, channel_depths, alpha=0.4)

    try:
        smoothed_firing_rate = savgol_filter(mean_firing_rate_by_depth["firing_rate"], 10, 2)
        ax_fr.plot(smoothed_firing_rate, mean_firing_rate_by_depth.index.tolist(), c="r")
    except Exception:
        logging.info("Smooting firing rates failed.")
    ax_fr.set_title("Unit Firing Rate By Depth")
    ax_fr.set_xlabel("Firing rate (Hz)")
    ax_fr.set_ylabel("Depth ($\\mu m$)")
    ax_fr.legend()
    ax_fr.spines[["top", "right"]].set_visible(False)

    ax_text = axs_yield[2, 2]
    ax_text.axis("off")

    metric_values = {
        "# units": number_of_units,
        "# SUA": number_of_sua_units,
        "# MUA": number_of_mua_units,
        "# noise": number_of_noise_units,
        "# passing default QC": number_of_good_units,
    }

    metric_values_str = None
    for metric_name, metric_value in metric_values.items():
        if metric_values_str is None:
            metric_values_str = f"{metric_name}: {metric_value}"
        else:
            metric_values_str += f"\n{metric_name}: {metric_value}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax_text.text(0, 1, metric_values_str, transform=ax_text.transAxes, fontsize=14, verticalalignment="top", bbox=props)

    fig_yield.suptitle("Unit Metrics Yield", fontsize=15)
    fig_yield.tight_layout()
    unit_yield_path = recording_fig_folder / "unit_yield.png"
    fig_yield.savefig(unit_yield_path, dpi=300)

    if relative_to is not None:
        unit_yield_path = unit_yield_path.relative_to(relative_to)

    sorting_summary_link = None
    if visualization_output is not None:
        if recording_name in visualization_output:
            sorting_summary_link = visualization_output[recording_name].get("sorting_summary")
            sorting_summary_str = f"[Sortingview]({sorting_summary_link})"
        else:
            sorting_summary_str = None
    else:
        sorting_summary_str = None

    value_with_options = {
        "value": "",
        "options": ["Good", "Low Yield", "High Noise"],
        "status": ["Pass", "Fail", "Fail"],
        "type": "dropdown",
    }
    value = DropdownMetric(**value_with_options)
    yield_metric_description = (
        "This metric displays a summary of spike sorted units, including distributions of key features "
        "and number of units with different labels. Use this metric to report a low yield of units or "
        "a disproportionate high number of bad units with respect to good units. "
    )
    if sorting_summary_str is not None:
        yield_metric_description += f"The sorting summary can also be viewed in {sorting_summary_str}"

    yield_metric = QCMetric(
        name=f"Unit Metrics Yield - {recording_name_abbrv}",
        modality=Modality.ECEPHYS,
        stage=Stage.PROCESSING,
        description=yield_metric_description,
        reference=str(unit_yield_path),
        value=value,
        status_history=[status_pending],
        tags={
            "probe": recording_name_abbrv
        }
    )
    metrics.append(yield_metric)

    logging.info("Generating FIRING RATE metric")
    num_segments = sorting_analyzer.get_num_segments()
    fig_fr, axs_fr = _get_fig_axs(num_segments, 1, subplot_figsize=(5, 3))
    spike_vector = sorting_analyzer.sorting.to_spike_vector()
    if sorting_analyzer.has_recording():
        recording = sorting_analyzer.recording
    else:
        recording = None

    for segment_index in range(num_segments):
        spike_vector_segment = spike_vector[spike_vector["segment_index"] == segment_index]

        if recording is not None:
            times = recording.get_times(segment_index=segment_index)
            spike_times = times[spike_vector_segment["sample_index"]]
        else:
            spike_times = spike_vector_segment["sample_index"] / sorting_analyzer.sampling_frequency

        duration = sorting_analyzer.get_num_samples(segment_index=segment_index) / sorting_analyzer.sampling_frequency
        num_bins = int(np.ceil(duration / bin_duration_hist_s))

        spike_times_hist, bin_edges = np.histogram(spike_times, bins=num_bins)
        bin_centers = bin_edges[:-1] + bin_duration_hist_s / 2
        axs_fr[segment_index, 0].plot(bin_centers, spike_times_hist)
        axs_fr[segment_index, 0].set_title(f"Population firing rate")
        axs_fr[segment_index, 0].set_xlabel("Time (s)")
        axs_fr[segment_index, 0].set_ylabel("Firing rate (Hz)")
        axs_fr[segment_index, 0].spines[["top", "right"]].set_visible(False)
        axs_fr[segment_index, 0].ticklabel_format(useOffset=False, style='plain', axis='x')

    fig_fr.tight_layout()
    firing_rate_path = recording_fig_folder / "firing_rate.png"
    fig_fr.savefig(firing_rate_path, dpi=300)
    if relative_to is not None:
        firing_rate_path = firing_rate_path.relative_to(relative_to)

    value_with_options = {
        "value": "",
        "options": ["No problems detected", "Seizure", "Firing Rate Gap"],
        "status": ["Pass", "Fail", "Fail"],
        "type": "dropdown",
    }
    value = DropdownMetric(**value_with_options)
    firing_rate_metric_description = (
        "This metric shows the population firing rate across the entire recording. Use this metric "
        "to spot abnormal seizure-like activity or firing rate gaps, which could result from hardware "
        "failures diring acquisition."
    )
    firing_rate_metric = QCMetric(
        name=f"Firing rate - {recording_name_abbrv}",
        modality=Modality.ECEPHYS,
        stage=Stage.PROCESSING,
        description=firing_rate_metric_description,
        reference=str(firing_rate_path),
        value=value,
        status_history=[status_pending],
        tags={
            "probe": recording_name_abbrv
        }
    )
    metrics.append(firing_rate_metric)

    if raw_recording is not None:
        # figure out whether ecephys_compressed or ecephys/ecephys_compressed #TODO
        recording_relative_path = get_recording_relative_path(raw_recording)
        if recording_relative_path is None:
            curation_link = None
        else:
            curation_link = "https://ephys.allenneuraldynamics.org/ephys_gui_app?analyzer_path={derived_asset_location}/postprocessed/"
            curation_link += f"{recording_name}.zarr&recording_path="
            curation_link += "{raw_asset_location}/"
            curation_link += recording_relative_path

    if curation_link is not None:
        logging.info("Generating SORTING CURATION metric")
        curation_link_url = quote(curation_link)
        sorting_curation_metric_description = (
            "This metric renders the SpikeInterface GUI through the AIND ephys portal. "
            "You can use the GUI to curate spike sorting results (label, remove, merge, split). "
            "Use the 'Submit to parent' button to submit the curation data to the QC Portal."
        )
        # Create default curation value from curation
        curation_dict = get_default_curation_value(sorting_analyzer)
        if curation_dict is not None:
            curation_value = [curation_dict]
            curation_history = [
                CurationHistory(
                    curator="Ephys pipeline",
                    timestamp=now,
                )
            ]
        else:
            logging.info("\tCurated dictionary is invalid.")
            curation_value = []
            curation_history = []

        sorting_curation_metric = CurationMetric(
            name=f"Sorting Curation - {recording_name_abbrv}",
            type="Spike sorting curation",
            value=curation_value,
            curation_history=curation_history,
            modality=Modality.ECEPHYS,
            stage=Stage.PROCESSING,
            description=sorting_curation_metric_description,
            reference=curation_link_url,
            status_history=[QCStatus(evaluator="", status=Status.PASS, timestamp=now)],
            tags={
                "probe": recording_name_abbrv
            }
        )
        metrics.append(sorting_curation_metric)

    return metrics


### EVENTS UTILS

# saturation values for different NP probes
saturation_thresholds_uv = {
    "PRB_1_4_0480_1": 0.6 / 500 * 1e6,
    "PRB_1_4_0480_1_C": 0.6 / 500 * 1e6,
    "PRB_1_2_0480_2": 0.6 / 500 * 1e6,
    "NP1010": 0.6 / 500 * 1e6,
    # NHP probes
    "NP1015": 0.6 / 500 * 1e6,
    "NP1016": 0.6 / 500 * 1e6,
    "NP1022": 0.6 / 500 * 1e6,
    "NP1030": 0.6 / 500 * 1e6,
    "NP1031": 0.6 / 500 * 1e6,
    "NP1032": 0.6 / 500 * 1e6,
    # NP2.0
    "NP2000": 0.5 / 80 * 1e6,
    "NP2010": 0.5 / 80 * 1e6,
    "NP2013": 0.62 / 100 * 1e6,
    "NP2014": 0.62 / 100 * 1e6,
    "NP2003": 0.62 / 100 * 1e6,
    "NP2004": 0.62 / 100 * 1e6,
    "PRB2_1_2_0640_0": 0.5 / 80 * 1e6,
    "PRB2_4_2_0640_0": 0.5 / 80 * 1e6,
    # Quad Base (same as NP2)
    "NP2020": 0.62 / 100 * 1e6,
    # Other probes
    "NP1100": 0.6 / 500 * 1e6,  # Ultra probe - 1 bank
    "NP1110": 0.6 / 500 * 1e6,  # Ultra probe - 16 banks
    "NP1121": 0.6 / 500 * 1e6,  # Ultra probe - beta configuration
    "NP1300": 0.6 / 500 * 1e6,  # Opto probe
}


def find_saturation_events(
    recording: si.BaseRecording,
    saturation_threshold_uv: float,
    exclude_sweep_ms: float = 1,
    radius_um: float = 200,
    **job_kwargs,
):
    """
    Find saturation events in a recording.

    Parameters
    ----------
    recording : BaseRecording
        The recording object.
    saturation_threshold_uv : float
        The saturation threshold in microvolts.
    exclude_sweep_ms : float, default: 1
        The time in milliseconds to exclude around the saturation event.
    radius_um : float, default: 200
        The radius in micrometers to consider for saturation events.
    job_kwargs : dict
        The job kwargs.

    Returns
    -------
    positive_saturation_events : np.ndarray
        The positive saturation events.
    negative_saturation_events : np.ndarray
        The negative saturation events.
    """
    from spikeinterface.core.node_pipeline import run_node_pipeline
    from spikeinterface.sortingcomponents.peak_detection.locally_exclusive import LocallyExclusivePeakDetector

    channel_distance = si.get_channel_distances(recording)
    neighbours_mask = channel_distance <= radius_um
    num_channels = recording.get_num_channels()

    # here we set absolute thresholds externally, since we know the saturation thresholds
    saturation_both = LocallyExclusivePeakDetector(recording, noise_levels=np.ones(num_channels))
    abs_thresholds = np.array([saturation_threshold_uv / recording.get_channel_gains()[0]] * num_channels)
    saturation_both.args = ("both", abs_thresholds, exclude_sweep_ms, neighbours_mask)

    job_name = f"finding saturation events"
    squeeze_output = True
    nodes = [saturation_both]
    outs = run_node_pipeline(
        recording,
        nodes,
        job_kwargs,
        job_name=job_name,
        squeeze_output=squeeze_output,
    )

    # only keep unique saturation events
    outs = outs[np.unique(outs["sample_index"], return_index=True)[1]]

    positive_saturation_events = outs[outs["amplitude"] > 0]
    negative_saturation_events = outs[outs["amplitude"] < 0]

    return positive_saturation_events, negative_saturation_events

import argparse
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np

HARTREE_TO_EV = 27.2114


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Dirac estimates and XML-derived band gaps across strains."
    )
    parser.add_argument(
        "--strains",
        type=float,
        nargs="+",
        required=True,
        help="Space-separated strain values (example: --strains 0.0 10.0 15.0).",
    )
    parser.add_argument(
        "--outdir",
        default="tmp",
        help="QE outdir containing PREFIX.save/data-file-schema.xml files (default: tmp).",
    )
    return parser.parse_args()


def read_checkpoint(strain):
    strain_str = f"{strain:.1f}"
    checkpoint_path = os.path.join("checkpoints", f"checkpoint{strain_str}.pkl")

    if not os.path.exists(checkpoint_path):
        return strain_str, checkpoint_path, None

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    return strain_str, checkpoint_path, checkpoint


def format_point(kx, ky):
    if kx is None or ky is None:
        return "N/A"
    return f"({kx:.8f}, {ky:.8f})"

def format_gap(gap):
    if gap is None:
        return "N/A"
    return f"{gap:.8f}"


def build_xml_path(outdir, prefix):
    return os.path.join(outdir, f"{prefix}.save", "data-file-schema.xml")


def parse_nelec(root):
    nelec_text = root.findtext(".//nelec")
    if nelec_text is None:
        return None
    try:
        return float(nelec_text)
    except ValueError:
        return None


def parse_fermi_ev(root):
    fermi_text = root.findtext(".//fermi_energy")
    if fermi_text is None:
        return None
    try:
        return float(fermi_text) * HARTREE_TO_EV
    except ValueError:
        return None


def infer_nocc_from_fermi(bands_ev, fermi_ev, tol_ev=1e-6):
    if fermi_ev is None:
        return None
    occupied_counts = np.sum(bands_ev <= (fermi_ev + tol_ev), axis=1)
    if occupied_counts.size == 0:
        return None
    # Use a robust typical occupancy across k-points.
    nocc = int(np.median(occupied_counts))
    if nocc <= 0 or nocc >= bands_ev.shape[1]:
        return None
    return nocc


def parse_xml_gaps(xml_path):
    if not os.path.exists(xml_path):
        return {"status": f"missing xml: {xml_path}"}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        return {"status": f"xml parse error: {exc}"}

    bands_rows = []
    k_points = []
    for ks in root.findall(".//ks_energies"):
        kpt_node = ks.find("k_point")
        eig_node = ks.find("eigenvalues")
        if kpt_node is None or eig_node is None or eig_node.text is None:
            continue

        try:
            eigs_ev = np.array([float(x) for x in eig_node.text.split()]) * HARTREE_TO_EV
            kpt = [float(x) for x in kpt_node.text.split()]
        except ValueError:
            continue

        if eigs_ev.size < 2 or len(kpt) < 2:
            continue

        bands_rows.append(eigs_ev)
        k_points.append(kpt[:2])

    if not bands_rows:
        return {"status": "no ks_energies/eigenvalues found"}

    bands_ev = np.vstack(bands_rows)  # shape (nk, nbnd)
    k_points = np.array(k_points)
    nbnd = bands_ev.shape[1]

    nelec = parse_nelec(root)
    if nelec is not None:
        nocc = int(round(nelec / 2.0))
    else:
        nocc = None

    if nocc is None or nocc <= 0 or nocc >= nbnd:
        nocc = infer_nocc_from_fermi(bands_ev, parse_fermi_ev(root))

    if nocc is None or nocc <= 0 or nocc >= nbnd:
        return {
            "status": "could not determine occupied-band index",
            "nbnd": nbnd,
            "nelec": nelec,
        }

    valence_ev = bands_ev[:, nocc - 1]
    conduction_ev = bands_ev[:, nocc]

    direct_gaps = conduction_ev - valence_ev
    direct_min_idx = int(np.argmin(direct_gaps))
    direct_min_gap = float(direct_gaps[direct_min_idx])
    direct_kx, direct_ky = k_points[direct_min_idx]

    vbm_idx = int(np.argmax(valence_ev))
    cbm_idx = int(np.argmin(conduction_ev))
    vbm_ev = float(valence_ev[vbm_idx])
    cbm_ev = float(conduction_ev[cbm_idx])
    fundamental_raw = cbm_ev - vbm_ev
    # Physical gap cannot be negative; negatives indicate overlap/semimetallic behavior.
    fundamental_gap = max(0.0, fundamental_raw)

    return {
        "status": "ok",
        "nocc": nocc,
        "direct_gap_ev": direct_min_gap,
        "direct_kx": float(direct_kx),
        "direct_ky": float(direct_ky),
        "fundamental_gap_ev": float(fundamental_gap),
        "fundamental_gap_raw_ev": float(fundamental_raw),
    }


def short_status(status):
    if status == "ok":
        return "ok"
    if status.startswith("missing xml:"):
        return "missing xml"
    if status.startswith("xml parse error:"):
        return "xml parse err"
    return "xml issue"


def main():
    args = parse_args()

    print("Dirac and XML gap summary")
    print("=" * 190)
    print(
        f"{'strain(%)':>10}  "
        f"{'coarse_dirac (kx, ky)':>30}  "
        f"{'zoom_dirac (kx, ky)':>30}  "
        f"{'chkpt_zoom_gap':>14}  "
        f"{'coarse_direct':>13}  "
        f"{'coarse_fund':>11}  "
        f"{'zoom_direct':>11}  "
        f"{'zoom_fund':>9}  "
        f"status"
    )
    print("-" * 230)

    for strain in args.strains:
        strain_str, checkpoint_path, checkpoint = read_checkpoint(strain)
        prefix = f"graphene{strain_str}"
        zoomprefix = f"graphene{strain_str}zoom"
        coarse_xml_path = build_xml_path(args.outdir, prefix)
        zoom_xml_path = build_xml_path(args.outdir, zoomprefix)
        coarse_xml = parse_xml_gaps(coarse_xml_path)
        zoom_xml = parse_xml_gaps(zoom_xml_path)

        if checkpoint is None:
            print(
                f"{strain_str:>10}  {'N/A':>30}  {'N/A':>30}  "
                f"{'N/A':>14}  "
                f"{format_gap(coarse_xml.get('direct_gap_ev')):>13}  "
                f"{format_gap(coarse_xml.get('fundamental_gap_ev')):>11}  "
                f"{format_gap(zoom_xml.get('direct_gap_ev')):>11}  "
                f"{format_gap(zoom_xml.get('fundamental_gap_ev')):>9}  "
                f"missing checkpoint"
            )
            continue

        coarse_kx = checkpoint.get("kx_dirac")
        coarse_ky = checkpoint.get("ky_dirac")
        zoom_kx = checkpoint.get("zoom_kx_dirac")
        zoom_ky = checkpoint.get("zoom_ky_dirac")
        zoom_gap = checkpoint.get("zoom_min_gap")

        coarse_point = format_point(coarse_kx, coarse_ky)
        zoom_point = format_point(zoom_kx, zoom_ky)
        zoom_gap_text = format_gap(zoom_gap)

        if coarse_point == "N/A" and zoom_point == "N/A":
            checkpoint_status = "no dirac keys"
        else:
            checkpoint_status = "ok"

        xml_flags = []
        if coarse_xml.get("status") != "ok":
            xml_flags.append(f"coarse:{short_status(coarse_xml['status'])}")
        if zoom_xml.get("status") != "ok":
            xml_flags.append(f"zoom:{short_status(zoom_xml['status'])}")

        status = checkpoint_status if not xml_flags else f"{checkpoint_status}; " + ", ".join(xml_flags)

        print(
            f"{strain_str:>10}  {coarse_point:>30}  {zoom_point:>30}  "
            f"{zoom_gap_text:>14}  "
            f"{format_gap(coarse_xml.get('direct_gap_ev')):>13}  "
            f"{format_gap(coarse_xml.get('fundamental_gap_ev')):>11}  "
            f"{format_gap(zoom_xml.get('direct_gap_ev')):>11}  "
            f"{format_gap(zoom_xml.get('fundamental_gap_ev')):>9}  "
            f"{status}"
        )

    print("=" * 230)
    print("Legend: direct=min_k[Ec(k)-Ev(k)], fund=max(0, min_k Ec - max_k Ev).")


if __name__ == "__main__":
    main()

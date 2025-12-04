"""
Audio Beamforming Example
==========================

This example demonstrates how to use the beamforming capabilities
for direction-of-arrival (DOA) estimation and audio source separation.

Features:
- Microphone array processing
- Delay-and-sum beamforming
- MUSIC algorithm for DOA estimation
- Real-time audio streaming
- Spatial audio filtering
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft

from aptt.model.beamforming.beamforming import DelayAndSumBeamformer, MUSICBeamformer
from aptt.model.doa import DOAEstimator
from aptt.utils.array_calibration import circular_array_geometry, linear_array_geometry
from aptt.utils.device import get_best_device


class AudioBeamformingPipeline:
    """Complete audio beamforming pipeline."""
    
    def __init__(
        self,
        n_mics: int = 8,
        array_radius: float = 0.1,
        sample_rate: int = 16000,
        array_type: str = 'circular',
        device: str | None = None
    ):
        self.n_mics = n_mics
        self.sample_rate = sample_rate
        self.device = device or str(get_best_device())
        
        print(f"🎙️  Audio Beamforming Configuration")
        print(f"   Microphones: {n_mics}")
        print(f"   Array type: {array_type}")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Device: {self.device}")
        
        # Create microphone array geometry
        if array_type == 'circular':
            self.mic_positions = circular_array_geometry(n_mics, array_radius)
        elif array_type == 'linear':
            self.mic_positions = linear_array_geometry(n_mics, array_radius * 2)
        else:
            raise ValueError(f"Unknown array type: {array_type}")
        
        # Initialize beamformers
        self.das_beamformer = DelayAndSumBeamformer(
            mic_positions=self.mic_positions,
            sample_rate=sample_rate
        ).to(self.device)
        
        self.music_beamformer = MUSICBeamformer(
            mic_positions=self.mic_positions,
            n_sources=2  # Estimate up to 2 sources
        ).to(self.device)
        
        self.doa_estimator = DOAEstimator(
            mic_positions=self.mic_positions,
            sample_rate=sample_rate
        ).to(self.device)
        
        print("✅ Beamformers initialized!")
    
    def load_audio(self, path: str) -> tuple[np.ndarray, int]:
        """Load multi-channel audio file."""
        sr, audio = wavfile.read(path)
        
        # Convert to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # Ensure correct shape: (n_samples, n_channels)
        if audio.ndim == 1:
            # Mono - replicate for all mics (simulation)
            audio = np.tile(audio[:, None], (1, self.n_mics))
        
        print(f"📂 Loaded audio: {path}")
        print(f"   Shape: {audio.shape}")
        print(f"   Duration: {len(audio) / sr:.2f}s")
        
        return audio, sr
    
    def estimate_doa(self, audio: np.ndarray, n_angles: int = 360) -> np.ndarray:
        """Estimate direction of arrival."""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio.T).float().unsqueeze(0).to(self.device)
        
        # Compute spatial spectrum
        with torch.no_grad():
            angles = torch.linspace(0, 360, n_angles).to(self.device)
            spectrum = self.doa_estimator(audio_tensor, angles)
        
        return spectrum.cpu().numpy()[0]
    
    def beamform_audio(
        self,
        audio: np.ndarray,
        target_angle: float,
        method: str = 'das'
    ) -> np.ndarray:
        """Apply beamforming to enhance audio from target direction."""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio.T).float().unsqueeze(0).to(self.device)
        
        # Apply beamforming
        with torch.no_grad():
            if method == 'das':
                output = self.das_beamformer(audio_tensor, target_angle)
            elif method == 'music':
                output = self.music_beamformer(audio_tensor, target_angle)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return output.cpu().numpy()[0, 0]
    
    def visualize_spatial_spectrum(
        self,
        spectrum: np.ndarray,
        save_path: str | None = None
    ):
        """Visualize spatial spectrum (DOA)."""
        angles = np.linspace(0, 360, len(spectrum))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Polar plot
        ax1 = plt.subplot(121, projection='polar')
        ax1.plot(np.deg2rad(angles), spectrum, linewidth=2)
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_title('Spatial Spectrum (Polar)', fontsize=12, pad=20)
        ax1.grid(True)
        
        # Cartesian plot
        ax2 = plt.subplot(122)
        ax2.plot(angles, spectrum, linewidth=2)
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Power')
        ax2.set_title('Spatial Spectrum (Cartesian)')
        ax2.grid(True)
        
        # Find peaks
        peaks = self._find_peaks(spectrum, angles)
        for angle, power in peaks:
            ax2.axvline(angle, color='r', linestyle='--', alpha=0.5)
            ax2.text(angle, power, f'{angle:.0f}°', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Saved plot to {save_path}")
        else:
            plt.show()
    
    def _find_peaks(
        self,
        spectrum: np.ndarray,
        angles: np.ndarray,
        n_peaks: int = 3
    ) -> list[tuple[float, float]]:
        """Find peaks in spatial spectrum."""
        from scipy.signal import find_peaks
        
        peaks_idx, properties = find_peaks(spectrum, height=np.max(spectrum) * 0.5)
        
        if len(peaks_idx) == 0:
            return []
        
        # Sort by height
        sorted_idx = np.argsort(properties['peak_heights'])[::-1][:n_peaks]
        peaks_idx = peaks_idx[sorted_idx]
        
        return [(angles[idx], spectrum[idx]) for idx in peaks_idx]


def demo_doa_estimation(args):
    """Demo: Estimate direction of arrival."""
    print("\n" + "=" * 60)
    print("🎯 DOA Estimation Demo")
    print("=" * 60)
    
    pipeline = AudioBeamformingPipeline(
        n_mics=args.n_mics,
        array_radius=args.array_radius,
        sample_rate=args.sample_rate,
        array_type=args.array_type,
        device=args.device
    )
    
    # Load audio
    audio, sr = pipeline.load_audio(args.input)
    
    # Estimate DOA
    print("\n🔍 Estimating direction of arrival...")
    spectrum = pipeline.estimate_doa(audio[:sr * 2])  # Use first 2 seconds
    
    # Find peaks
    angles = np.linspace(0, 360, len(spectrum))
    peaks = pipeline._find_peaks(spectrum, angles)
    
    print(f"\n📍 Detected {len(peaks)} source(s):")
    for i, (angle, power) in enumerate(peaks, 1):
        print(f"   Source {i}: {angle:.1f}° (power: {power:.2f})")
    
    # Visualize
    output_path = Path(args.output) / 'doa_spectrum.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.visualize_spatial_spectrum(spectrum, str(output_path))


def demo_beamforming(args):
    """Demo: Apply beamforming to extract audio from specific direction."""
    print("\n" + "=" * 60)
    print("🎯 Beamforming Demo")
    print("=" * 60)
    
    pipeline = AudioBeamformingPipeline(
        n_mics=args.n_mics,
        array_radius=args.array_radius,
        sample_rate=args.sample_rate,
        array_type=args.array_type,
        device=args.device
    )
    
    # Load audio
    audio, sr = pipeline.load_audio(args.input)
    
    # Apply beamforming
    print(f"\n🎚️  Applying {args.method.upper()} beamforming...")
    print(f"   Target angle: {args.angle}°")
    
    output_audio = pipeline.beamform_audio(audio, args.angle, args.method)
    
    # Save output
    output_path = Path(args.output) / f'beamformed_{args.angle}deg.wav'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert back to int16
    output_audio_int = (output_audio * 32767).astype(np.int16)
    wavfile.write(str(output_path), sr, output_audio_int)
    
    print(f"✅ Saved beamformed audio to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='APTT Audio Beamforming Example')
    parser.add_argument('--mode', type=str, default='doa', choices=['doa', 'beamform'],
                        help='Demo mode')
    parser.add_argument('--input', type=str, required=True,
                        help='Input audio file (multi-channel WAV)')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--n-mics', type=int, default=8,
                        help='Number of microphones')
    parser.add_argument('--array-radius', type=float, default=0.1,
                        help='Array radius in meters')
    parser.add_argument('--array-type', type=str, default='circular',
                        choices=['circular', 'linear'],
                        help='Microphone array geometry')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--angle', type=float, default=0.0,
                        help='Target angle for beamforming (degrees)')
    parser.add_argument('--method', type=str, default='das',
                        choices=['das', 'music'],
                        help='Beamforming method')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    print("🎵 APTT Audio Beamforming Example")
    print("=" * 60)
    
    if args.mode == 'doa':
        demo_doa_estimation(args)
    else:
        demo_beamforming(args)


if __name__ == '__main__':
    main()

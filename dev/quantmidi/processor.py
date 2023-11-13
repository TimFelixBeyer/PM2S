




class QuantMIDIProcessor():
    def __init__(self, device='cuda'):
        self.device = device

    def process(self, performance_midi, quantised_midi):
        """
        Processes the performance midi and quantises it to the quantised midi.
        
        Args:
            performance_midi: filename to the performance midi.
            quantised_midi: filename to the quantised midi.
        """

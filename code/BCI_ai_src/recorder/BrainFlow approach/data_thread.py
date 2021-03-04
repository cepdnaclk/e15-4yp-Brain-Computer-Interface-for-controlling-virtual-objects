# -*- coding: utf-8 -*-
import time
import numpy as np
import threading
import pandas as pd

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter


class DataThread(threading.Thread):

    def __init__(self, board, board_id):
        threading.Thread.__init__(self)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.time_stamp_channel = BoardShim.get_timestamp_channel(board_id)

        self.keep_alive = True
        self.board = board

    def run(self):
        window_size = 5
        points_per_update = window_size * self.sampling_rate
        channel_datas = []

        while self.keep_alive:
            # data = self.board.get_board_data()
            time.sleep(1)
            # data = self.board.get_current_board_data(int(points_per_update))
            data = self.board.get_current_board_data(DataFilter.get_nearest_power_of_two(self.sampling_rate))
            df = pd.DataFrame(np.transpose(data))[self.eeg_channels]
            channel_datas.append(np.transpose(data)[:, 1:9])
            print(df.shape)

            # for channel in self.eeg_channels:
            #     print('channel : ',channel," shape ",data[channel].shape)

        print(np.array(channel_datas).shape)
        # np.save(channel_datas,f"{int(time.time())}.csv",'w')


def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = 'COM11'
    board_id = BoardIds.CYTON_BOARD.value

    board = BoardShim(board_id, params)
    board.prepare_session()

    board.start_stream(45000, 'file://cyton_data.csv:w')
    time.sleep(6)

    data_thread = DataThread(board, board_id)
    data_thread.start()

    try:
        time.sleep(10)
    finally:
        data_thread.keep_alive = False;
        data_thread.join()

    board.stop_stream()
    board.release_session()


if __name__ == "__main__":
    main()

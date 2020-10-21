import pickle
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from scipy.io import loadmat
from scipy.cluster.vq import kmeans

class RingachStim():
    
    def __init__(self):
        pass

    @classmethod
    def whiten_frames(cls, frames_grey, n_chunks=10):
        '''
        Whiten frames in chunks
        '''
        print('Whitening frames', end='')
        new_frames = np.zeros(frames_grey.shape)
        n_frames = frames_grey.shape[0]
        chunk_size = int(np.ceil(n_frames/n_chunks))

        for chunk_idx in np.arange(0, n_frames, chunk_size):
            print('.', end='')
            chunk = frames_grey[chunk_idx:chunk_idx+chunk_size,:,:]
            ft = np.fft.fft2(chunk)
            ft = ft / np.abs(ft)
            ft[:, 0, 0] = 0 # get rid of DC
            new_frames[chunk_idx:chunk_idx+chunk_size,:,:] = np.real(np.fft.ifft2(ft))
        print('done')
        return new_frames

    def get_movie_frame_idx(self, movie_idx, segment_idx, frame_idx):
        '''
        Get movie frame from original jpeg file
        '''
        path = Path(self.data_dir, 'pvc-1', 'crcns-ringach-data', 'movie_frames', 
                    'movie%03d_%03d.images/movie%03d_%03d_%03d.jpeg' % \
                    (movie_idx, segment_idx, movie_idx, segment_idx, frame_idx))
        im = image.imread(path)
        im = np.mean(im, axis=2)
        return im
    
class RingachStimPreloaded(RingachStim):
    
    def __init__(self, downsample=2, whitened=False, data_dir='.'):
        self.data_dir = data_dir
        self.frames = None
        self.downsample = downsample
        self.whitened = whitened
    
    def load(self):
        '''
        Get all movie frames
        '''
        type_str = 'whitened' if self.whitened else 'movie'
        path = Path(".", "%s_frames_d%d.pkl" % (type_str, self.downsample))

        if path.exists():
            print('Loading from %s' % (str(path)))
            self.frames = pickle.load(open(path, "rb"))
        else:
            print('Loading from jpegs')
            self.frames = self.get_all_frames_from_jpeg()

    def get_all_frames_from_jpeg(self):
        '''
        Get movie frames from jpeg and save to *_frames_d*.pkl
        '''
        frames = []
        movie_idx = 0
        segment_idx = 0
        frame_idx = 0
        movie_frames = []
        for movie_idx in range(4):
            print('Loading movie %d' % movie_idx)
            segments = []
            for segment_idx in range(30):
                print('Loading segment %d' % segment_idx, end='')
                frames = []
                for frame_idx in range(2000):
                    try:
                        frame = self.get_movie_frame_idx(movie_idx, segment_idx, frame_idx)
                    except FileNotFoundError:
                        print('...got %d frames' % (frame_idx))
                        break
                    if self.downsample>1:
                        frame = frame[::self.downsample, ::self.downsample]
                    frames.append(frame)
                if self.whitened:
                    frames = self.whiten_frames(np.stack(frames, axis=0))
                else:
                    frames = np.stack(frames, axis=0)
                segments.append(frames)
            movie_frames.append(segments)

        type_str = 'whitened' if self.whitened else 'movie'
        path = Path(".", "%s_frames_d%d.pkl" % (type_str, self.downsample))
        print('Dumping to %s' % str(path))
        pickle.dump(movie_frames, open(path, "wb"))
        
    def get_frames(self, framelist, offset=0):
        frames = []
        for (movie_idx, segment_idx, frame_idxes) in framelist:
            frame_idxes = frame_idxes + offset

            segment_len = self.frames[movie_idx][segment_idx].shape[0]
            frame_idxes = frame_idxes[np.where(frame_idxes>0)[0]]
            frame_idxes = frame_idxes[np.where(frame_idxes<segment_len)[0]]
            frames.append(self.frames[movie_idx][segment_idx][frame_idxes, :])
        return np.concatenate(frames, axis=0)

class RingachStimPiecemeal(RingachStim):

    def __init__(self, downsample=2, whitened=False, data_dir='.'):
        self.data_dir = '.'
        self.frames = None
        self.downsample = downsample
        self.whitened = whitened
    
    def get_pixels(self, framelist, y_min=0, x_min=0, orig_size=None, offset=0):
        '''
        Get frames from a certain movie segment, cropping and downsampling if desired
        '''
        frames = []
        for (movie_idx, segment_idx, frame_idxes) in framelist:
            frame_idxes = frame_idxes + offset
            
            for frame_idx in frame_idxes:
                try:
                    frame = self.get_movie_frame_idx(movie_idx, segment_idx, frame_idx)
                except FileNotFoundError:
                    continue
                if orig_size is not None:
                    frame = frame[y_min:y_min+orig_size, x_min:x_min+orig_size]
                if self.downsample>1:
                    frame = frame[::self.downsample, ::self.downsample]
                frames.append(frame)
        frames = np.stack(frames, axis=0)

        if self.whitened:
            frames = self.whiten_frames(frames)
        return frames

def unpack(array):
    while True:
        if isinstance(array, np.ndarray):
            if any([s>1 for s in array.shape]):
                return array.squeeze()
            array = array[0]
        else:
            return array
        
def show(array, show_values=False):
    for field in array.dtype.names:
        if show_values:
            print(field, str(array[field]))
        else:
            print(field)

class RingachData():
    
    def __init__(self, dataset, channel_idx, spike_sort=True, data_dir='.'):
        self.data_dir = data_dir

        if isinstance(dataset, str):
            self.filename = dataset
        else:
            self.filename = self.list_files(disp=False)[dataset]
        print('Loading from %s' % self.filename)
        self.channel_idx = channel_idx
        self.load_data()
        if spike_sort:
            self.cluster_spikes()

    def list_files(self, disp=True):
        '''
        Get list of data files
        '''
        from pathlib import Path
        data_dir = Path(self.data_dir, 'pvc-1', 'crcns-ringach-data', 'neurodata')

        data_files = []
        for subdir in sorted(data_dir.iterdir()):
            for file in sorted(subdir.iterdir()):
                data_files.append(file)
                if disp:
                    print(file)
        return data_files

    def open_file(self):
        '''
        Open file and return data and n_channels
        '''
        pepANA = loadmat(self.filename)['pepANA']
        n_channels = len(unpack(unpack(pepANA)['elec_list']))
        return pepANA, n_channels
    
    def load_data(self):
        '''
        Get spike data for one channel
        '''
        pepANA, n_channels = self.open_file()
        print('Extracting spike data')
        lor = unpack(unpack(pepANA)['listOfResults'])
        n_conditions = lor.shape[0]
        conditions = []

        for condition_idx in range(n_conditions):
            print('.', end='')
            res = lor[condition_idx]
            condition_params = unpack(res['condition'])
            n_reps = unpack(res['noRepeats'])
            spike_times = []
            spike_shapes = []
            rep_numbers = []
            reps = unpack(res['repeat'])

            sym = [unpack(s) for s in unpack(res['symbols'])]

            if 'movie_id' not in sym:
                print('Condition %d is not a movie stimulus' % condition_idx)
                continue

            val = [unpack(s) for s in unpack(res['values'])]

            for rep_idx in range(n_reps):
                if n_reps == 1:
                    rep = unpack(reps)
                else:
                    rep = unpack(reps[rep_idx])
                dat = unpack(unpack(rep['data'])[self.channel_idx])
                spike_times.append(unpack(dat[0]))
                spike_shapes.append(unpack(dat[1]))
                rep_numbers.append(np.array([rep_idx]*spike_times[-1].shape[0]))

            conditions.append({'condition_params': condition_params,
                         'symbols': sym,
                         'values': val,
                         'n_reps': n_reps,
                         'spike_times': np.concatenate(spike_times),
                         'spike_shapes': np.concatenate(spike_shapes, axis=1),
                         'rep_numbers': np.concatenate(rep_numbers)})

        print('done')
        self.electrode_data = {'n_conditions': n_conditions,
                               'conditions': conditions}

    def cluster_spikes(self, plot=False):
        print('Clustering spikes')

        all_spike_times = np.concatenate([c['spike_times'] for c in self.electrode_data['conditions']])
        all_spike_shapes = np.concatenate([c['spike_shapes'] for c in self.electrode_data['conditions']], axis=1)
        n_spikes = all_spike_shapes.shape[1]
        codebook, _ = kmeans(all_spike_shapes.astype(np.double).transpose(), 2)
        norm = np.sum(codebook**2, axis=1)

        # 0th cluster is noise, 1st cluster is spikes
        if norm[1]<norm[0]:
            codebook = codebook[::-1, :]

        for i, c in enumerate(self.electrode_data['conditions']):
            print('.', end='')
            spike_times = []
            spike_shapes = []
            rep_numbers = []
            for idx in range(c['spike_times'].shape[0]):
                time = c['spike_times'][idx]
                shape = c['spike_shapes'][:, idx]
                rep = c['spike_times'][idx]
                d0 = np.sum((shape-codebook[0])**2)
                d1 = np.sum((shape-codebook[1])**2)
                if d1 < d0:
                    spike_times.append(time)
                    spike_shapes.append(shape)
                    rep_numbers.append(rep)
            if len(spike_times) == 0:
                self.electrode_data['conditions'][i]['spike_times'] = np.array([])
                self.electrode_data['conditions'][i]['spike_shapes'] = np.array([])
                self.electrode_data['conditions'][i]['rep_numbers'] = np.array([])
            self.electrode_data['conditions'][i]['spike_times'] = np.array(spike_times)
            if self.electrode_data['conditions'][i]['spike_times'].shape[0] == 0:
                self.electrode_data['conditions'][i]['spike_shapes'] = \
                    np.zeros((all_spike_shapes.shape[0], 0))
            else:
                self.electrode_data['conditions'][i]['spike_shapes'] = np.stack(spike_shapes, axis=1)
            self.electrode_data['conditions'][i]['rep_numbers'] = np.array(rep_numbers)

        assignment = np.zeros(n_spikes)
        for i in range(n_spikes):
            d0 = np.sum((all_spike_shapes[:, i]-codebook[0])**2)
            d1 = np.sum((all_spike_shapes[:, i]-codebook[1])**2)
            assignment[i] = 0 if d0<d1 else 1

        spike_times = all_spike_times[np.where(assignment==1)[0]]

        if plot:
            plt.plot(all_spike_shapes[:,np.where(assignment==0)[0]], 'tab:gray', linewidth=0.1)
            plt.plot(all_spike_shapes[:,np.where(assignment==1)[0]],'b', linewidth=0.1)
            plt.plot(codebook.T,'r')

        print('done')

    def get_spike_frames(self, offset=0):
        '''
        Return [(movie_idx, segment_idx, frame_idxes), ..] on which spikes occurred
        '''
        data = []
        for condition in self.electrode_data['conditions']:
            if condition['symbols'] != ['movie_id', 'segment_id']:
                raise ValueError('Unexpected parameters, not movie_id and segment_id')
            frame_idxes = (np.floor(condition['spike_times']/3*90) + offset).astype(np.int)
            data.append((condition['values'][0], condition['values'][1], frame_idxes))
        return data
    
def get_stas_sqdevs(data, stim, offsets):
    '''
    Get STAS and sum of squared deviations for a range of time offsets
    '''
    stas = []
    sta_sqdevs = []
    print('Getting STAs', end='')
    for offset in offsets:
        print('.', end='')
        frame_idxes = data.get_spike_frames(offset=offset)
        frames = stim.get_frames(frame_idxes)
        stas.append(np.sum(frames, axis=0).squeeze())
        sta_sqdevs.append(np.sum(frames**2, axis=0).squeeze())
    print('done')
    return stas, sta_sqdevs

def plot_stas_sqdevs(dataset_idx, channel_idx, offsets, stas, sta_sqdevs):
    '''
    Plot STAs and STASQDEVs
    '''

    # subtract off mean of each frame
    sta_norms = [m - np.mean(m) for m in stas]
    # scale frames uniformly and make them fit in colormap
    norm = np.std(np.stack(sta_norms))*2
    sta_norms = [m / norm for m in sta_norms]
    # [print(np.min(a), np.max(a)) for a in sta_norms]

    plt.subplots(2, len(stas), figsize=(20, 3))
    for i in range(len(sta_norms)):
        plt.subplot(2, len(stas), i+1)
        ax = plt.imshow(sta_norms[i][4:-4,4:-4], clim=(-1, 1), cmap='seismic')
        plt.tick_params(labelbottom=False, labelleft=False)
        if i==0:
            plt.ylabel('STA')
            plt.xlabel('Unit %d,%d' % (dataset_idx, channel_idx))
    mn_sqdevs = np.sum(np.stack(sta_sqdevs), axis=0)
    sqdev_norms = [m / mn_sqdevs for m in sta_sqdevs]

    sqdev_norms = [m - np.mean(m[2:-2,2:-2]) for m in sqdev_norms]
    norm = np.std(np.stack(sqdev_norms))*8
    sqdev_norms = [m / norm for m in sqdev_norms]

    for i in range(len(sta_sqdevs)):
        plt.subplot(2, len(sta_sqdevs), len(stas)+i+1)
        ax = plt.imshow(sqdev_norms[i][2:-2,2:-2], clim=(-1, 1), cmap='seismic')
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.xlabel('%d ms' % (int(offsets[i]*3/90*1000)))
        if i==0:
            plt.ylabel('STSQDEV')

class FigViewer():

    def __init__(self, fig_dir='.'):
        self.fig_dir = Path(fig_dir)
        self.file_list = None
        self.get_file_list()
        self.file_idx = 0
        self.output = widgets.Output()
        self.button = widgets.Button(description='Next')
        self.button.on_click(self.on_button_clicked)
        display(self.output)
        display(self.button)
        self.show()

    def get_file_list(self):
        suffixes = ('.svg', 'jpg')
        files = list(Path(self.fig_dir).iterdir())
        files.sort()
        files = [f for f in files if (f.suffix == '.svg')]
        self.file_list = [f for f in files if f.suffix in suffixes]

        def show(self):
            with self.output:
                display(HTML(open(self.file_list[self.file_idx]).read()))
                clear_output(wait=True)
                self.file_idx = self.file_idx + 1

    def on_button_clicked(self, _):
        self.show()

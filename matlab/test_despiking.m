%% LOAD LFP

basepath = cd;
[~,basename] = fileparts(basepath);
datFile = fullfile(basepath,[basename,'.dat'])

nbChan = 96;
Fs = 20000;

opts.displaylevel = 0;

chunk = 6e6; %chunk size, 1e7 needs a lot of RAM & GPU memory (depending on the number of reference channel for the median)
refChan = [1:nbChan];
% highFc  = [300 8000]/10000; %default, assuming sampling frequency of 20kHz

infoFile = dir(datFile);
nbChunks = floor(infoFile.bytes/(nbChan*chunk*2));
Duration = infoFile.bytes/2/nbChan/Fs;
warning off
if nbChunks==0
    chunk = infoFile.bytes/(nbChan*2);
end

dat = memmapfile(datFile,'Format','int16','writable',true);

if exist([basename '_despike.dat'], 'file')
    delete([basename '_despike.dat'])
end
fidout = fopen([basename '_despike.dat'], 'a');

%% LOADING SPIKEs
%par = LoadXml([basename '.xml']);
[T,G,Map,Par]=LoadCluRes(basename);
shank_with_spk = unique(Map(:,2));


%% MAIN LOOP
for ix=0:nbChunks   
   fprintf('%i %i\n', ix, nbChunks);
    % load chunk of data
    idx = ix*chunk*nbChan+1;
    if ix<nbChunks
        m = dat.Data(idx:(ix+1)*chunk*nbChan);
    else
        m = dat.Data(idx:end);
        chunk = infoFile.bytes/(2*nbChan)-nbChunks*chunk;
        datF = reshape(m,[nbChan chunk]);
    end
    
    datF = reshape(m,[nbChan chunk]);
    datF = double(datF);
       
    % load only shanks with spikes
    datO = datF;
    for sh=shank_with_spk'                
        idx_neurons = Map(Map(:,2)==sh);
        filename = [basename '.spk.' int2str(sh)];
        wf = LoadSpikeWaveF(filename, length(Par.SpkGrps(sh).Channels), 40, Map(idx_neurons,3)');
        % restrict by waveforms
        best_channels = zeros(length(idx_neurons),2);
        
        for n=1:length(idx_neurons)
            [v, idxch] = sort(mean(abs(wf(:,15:25,n))'));
            best_channels(n,:) = Par.SpkGrps(sh).Channels(idxch(end-1:end));
        end
        
        channels_TODO = unique(best_channels);
        
        S = zeros(length(datF),length(idx_neurons));
        subT = T(((T>ix*chunk) .* (T<= ix*chunk+length(datF)))==1);
        subG = G(((T>ix*chunk) .* (T<= ix*chunk+length(datF)))==1);
        subT = subT - ix*chunk;
        subT(subT<=20) = 21;
        for n=1:length(idx_neurons)
           S(subT(subG==idx_neurons(n))-20,n) = 1;
        end
%         spktimes = find(S(:,1));
%         for ch=Par.ElecGp{sh}
        for j=1:length(channels_TODO)
            ch=channels_TODO(j);
            wb = datF(ch+1,:)';
            g = fitLFPpowerSpectrum(wb, .01,1000,Fs);
%             taking only S for best channels
            subS = S(:,sum((best_channels==ch)')>0);
            results = despikeLFP(wb, subS, eye(60), g, opts);
            %plot((1:length(wb))/Fs,wb,(1:length(wb))/Fs,results.z+10,...
                %spktimes/Fs,1000*ones(length(spktimes),1),'.')            
            datO(ch+1,:) = results.z;
        end
    end
    datO = int16(datO);
    fwrite(fidout,datO(:),'int16');
end



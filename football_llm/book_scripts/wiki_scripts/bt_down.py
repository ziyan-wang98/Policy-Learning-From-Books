import os
import libtorrent as lt
import time
import datetime

ses = lt.session()
ses.listen_on(6881, 6891)
params = {
    'save_path': os.environ.get('PLFB_WIKI_DUMP_DIR', './data/wiki_dump/07'),
    #By default Torrent File would be saved in a folder "Torrent" in selected Drive
    'storage_mode': lt.storage_mode_t(2), # Use Sparse Storage Mode
    }
# Magnet Link Goes Here
link = "magnet:?xt=urn:btih:P5HHEIRYV4K7J4FNGYKOTFEQJVPPMWO5&dn=enwiki-20230701-pages-articles-multistream&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
print(link)

handle = lt.add_magnet_uri(ses, link, params) # Adding the Torrent File
ses.start_dht()

begin = time.time()
print(datetime.datetime.now())

print ('Downloading Metadata...')
while (not handle.has_metadata()):
    time.sleep(1)
print ('MataData Imported, Attempting Download')

print("Starting, PLease Wait\n", handle.name())

# Print the various Information regarding Torrent
while (handle.status().state != lt.torrent_status.seeding):
    s = handle.status()
    state_str = ['queued', 'checking', 'downloading metadata', \
            'downloading', 'finished', 'seeding', 'allocating']
    print ('%.2f%% complete (down: %.1f kb/s up: %.1f kB/s peers: %d) %s ' % \
            (s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000, \
            s.num_peers, state_str[s.state]))
    time.sleep(5)

end = time.time()
print(handle.name(), "COMPLETE, Go to the Drive to Find the Downloaded File")

# Compute and Print the Time Elapsed
print("Elapsed Time: ",int((end-begin)//60),"min :", int((end-begin)%60), "sec")
print("Closing Session, Come Back Again")
print(datetime.datetime.now())

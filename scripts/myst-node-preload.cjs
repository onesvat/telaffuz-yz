const os = require('node:os');
const net = require('node:net');

try {
  os.networkInterfaces();
} catch {
  os.networkInterfaces = () => ({
    lo: [
      {
        address: '127.0.0.1',
        netmask: '255.0.0.0',
        family: 'IPv4',
        mac: '00:00:00:00:00:00',
        internal: true,
        cidr: '127.0.0.1/8',
      },
    ],
  });
}

const originalListen = net.Server.prototype.listen;

net.Server.prototype.listen = function listenOnLoopback(...args) {
  if (args[0] && typeof args[0] === 'object') {
    const options = { ...args[0] };
    if (!options.host || options.host === '0.0.0.0') options.host = '127.0.0.1';
    args[0] = options;
  } else if (args[1] === '0.0.0.0') {
    args[1] = '127.0.0.1';
  }
  return originalListen.apply(this, args);
};

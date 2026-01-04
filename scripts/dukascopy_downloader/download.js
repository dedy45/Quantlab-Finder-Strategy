/**
 * Dukascopy Data Downloader
 * 
 * Download historical OHLCV data from Dukascopy for Quant Lab.
 * Output: CSV files compatible with convert_csv_to_parquet.py
 * 
 * Usage:
 *   node download.js --symbol XAUUSD --timeframe h1 --start 2015-01-01 --end 2025-12-31
 *   node download.js --symbol EURUSD --timeframe d1 --years 10
 */

const { getHistoricalRates } = require('dukascopy-node');
const fs = require('fs');
const path = require('path');

// Parse command line arguments
function parseArgs() {
    const args = process.argv.slice(2);
    const config = {
        symbol: 'xauusd',
        timeframe: 'h1',
        start: null,
        end: null,
        years: 10,
        output: '../../data/raw'
    };

    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--symbol':
            case '-s':
                config.symbol = args[++i].toLowerCase();
                break;
            case '--timeframe':
            case '-t':
                config.timeframe = args[++i].toLowerCase();
                break;
            case '--start':
                config.start = args[++i];
                break;
            case '--end':
                config.end = args[++i];
                break;
            case '--years':
            case '-y':
                config.years = parseInt(args[++i]);
                break;
            case '--output':
            case '-o':
                config.output = args[++i];
                break;
            case '--help':
            case '-h':
                printHelp();
                process.exit(0);
        }
    }

    // Calculate dates if not provided
    if (!config.end) {
        config.end = new Date().toISOString().split('T')[0];
    }
    if (!config.start) {
        const endDate = new Date(config.end);
        const startDate = new Date(endDate);
        startDate.setFullYear(startDate.getFullYear() - config.years);
        config.start = startDate.toISOString().split('T')[0];
    }

    return config;
}

function printHelp() {
    console.log(`
Dukascopy Data Downloader for Quant Lab

Usage:
  node download.js [options]

Options:
  --symbol, -s     Symbol to download (default: xauusd)
  --timeframe, -t  Timeframe: m1, m5, m15, m30, h1, h4, d1 (default: h1)
  --start          Start date YYYY-MM-DD (default: 10 years ago)
  --end            End date YYYY-MM-DD (default: today)
  --years, -y      Years of data to download (default: 10)
  --output, -o     Output directory (default: ../../data/raw)
  --help, -h       Show this help

Examples:
  node download.js --symbol xauusd --timeframe h1 --years 10
  node download.js --symbol eurusd --timeframe d1 --start 2020-01-01 --end 2025-01-01

Available Symbols:
  Forex: eurusd, gbpusd, usdjpy, usdchf, audusd, usdcad, nzdusd
  Metals: xauusd (gold), xagusd (silver)
  Indices: spxusd, nsxusd, etxeur, jpxjpy
  Crypto: btcusd, ethusd
`);
}

// Map timeframe to dukascopy format
function mapTimeframe(tf) {
    const mapping = {
        'm1': 'm1',
        'm5': 'm5',
        'm15': 'm15',
        'm30': 'm30',
        'h1': 'h1',
        '1h': 'h1',
        'h4': 'h4',
        '4h': 'h4',
        'd1': 'd1',
        '1d': 'd1'
    };
    return mapping[tf.toLowerCase()] || 'h1';
}

// Convert data to CSV format compatible with convert_csv_to_parquet.py
function toCSV(data, symbol) {
    const header = 'Symbol,DateTime,Open,High,Low,Close,Volume';
    const rows = data.map(bar => {
        const dt = new Date(bar.timestamp);
        const dateStr = dt.toISOString().replace('T', ' ').replace('Z', '').slice(0, 19);
        return `${symbol.toUpperCase()},${dateStr},${bar.open},${bar.high},${bar.low},${bar.close},${bar.volume || 0}`;
    });
    return [header, ...rows].join('\n');
}

async function download(config) {
    console.log('============================================================');
    console.log('DUKASCOPY DATA DOWNLOADER');
    console.log('============================================================');
    console.log(`Symbol: ${config.symbol.toUpperCase()}`);
    console.log(`Timeframe: ${config.timeframe}`);
    console.log(`Period: ${config.start} to ${config.end}`);
    console.log('');

    const startTime = Date.now();

    try {
        console.log('[INFO] Downloading data from Dukascopy...');
        
        const data = await getHistoricalRates({
            instrument: config.symbol,
            dates: {
                from: new Date(config.start),
                to: new Date(config.end)
            },
            timeframe: mapTimeframe(config.timeframe),
            priceType: 'bid',
            volumes: true
        });

        console.log(`[OK] Downloaded ${data.length.toLocaleString()} bars`);

        if (data.length === 0) {
            console.log('[WARN] No data received');
            return;
        }

        // Show date range
        const firstBar = new Date(data[0].timestamp);
        const lastBar = new Date(data[data.length - 1].timestamp);
        console.log(`[INFO] Date range: ${firstBar.toISOString().split('T')[0]} to ${lastBar.toISOString().split('T')[0]}`);

        // Convert to CSV
        const csv = toCSV(data, config.symbol);

        // Ensure output directory exists
        const outputDir = path.resolve(__dirname, config.output);
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        // Save file
        const filename = `${config.symbol.toUpperCase()}_dukascopy_${config.timeframe.toUpperCase()}.csv`;
        const filepath = path.join(outputDir, filename);
        fs.writeFileSync(filepath, csv);

        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        const fileSize = (fs.statSync(filepath).size / 1024 / 1024).toFixed(2);

        console.log('');
        console.log('[OK] Download complete!');
        console.log(`  File: ${filepath}`);
        console.log(`  Bars: ${data.length.toLocaleString()}`);
        console.log(`  Size: ${fileSize} MB`);
        console.log(`  Time: ${elapsed}s`);
        console.log('');
        console.log('Next step: python scripts/convert_csv_to_parquet.py --file data/raw/' + filename);

    } catch (error) {
        console.error(`[FAIL] Download failed: ${error.message}`);
        process.exit(1);
    }
}

// Run
const config = parseArgs();
download(config);
